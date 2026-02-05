import logging
from pathlib import Path
from typing import List, Dict
from modules.translator import BaseTranslator
from utils.srt_handler import SRTHandler

logger = logging.getLogger(__name__)

class HybridRefiner(BaseTranslator):
    def __init__(self, s1_dir: str, l1_dir: str, mt_dir: str, output_dir: str, provider, config: Dict):
        # S1 (Original Transcription) is the master reference
        super().__init__(s1_dir, output_dir, bot=provider)
        self.l1_dir = Path(l1_dir)
        self.mt_dir = Path(mt_dir)
        self.config = config
        self.bot = provider
        self.name = "Hybrid Refiner (Triple-Source Arbitration)"

    def process_file(self, s1_file: Path):
        """Processes a single file by arbitrating between S1, L1, and Mt sources."""
        output_file = self.get_output_path(s1_file, ".srt")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Locate corresponding L1 and Mt files using relative path (supports subdirectories)
        relative_path = s1_file.relative_to(self.input_dir)
        l1_file = self.l1_dir / relative_path
        mt_file = self.mt_dir / relative_path

        if not l1_file.exists() or not mt_file.exists():
            logger.error(f"Missing input streams for {s1_file.name}. L1: {l1_file.exists()}, Mt: {mt_file.exists()}")
            return

        logger.info(f"Starting triple-source refinement for: {s1_file.name}")

        # Read all three streams
        try:
            with open(s1_file, "r", encoding="utf-8") as f: s1_raw = f.read()
            with open(l1_file, "r", encoding="utf-8") as f: l1_raw = f.read()
            with open(mt_file, "r", encoding="utf-8") as f: mt_raw = f.read()
        except Exception as e:
            logger.error(f"Failed to read input files for refinement: {e}")
            return

        # Core arbitration logic
        final_srt_content = self.refine_logic(s1_raw, l1_raw, mt_raw)
        
        # Final save with SRT standardization
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(SRTHandler.standardize(final_srt_content))
            self.wait_for_stability(output_file)
            logger.info(f"Successfully refined and saved: {output_file.name}")
        except Exception as e:
            logger.error(f"Failed to save refined SRT: {e}")

    def refine_logic(self, s1_text: str, l1_text: str, mt_text: str) -> str:
        """Slices the SRT into windows and performs the arbitration via LLM."""
        s1_blocks = SRTHandler.parse_to_blocks(s1_text)
        l1_blocks = SRTHandler.parse_to_blocks(l1_text)
        mt_blocks = SRTHandler.parse_to_blocks(mt_text)
        
        # Load the system protocol
        protocol_path = Path(self.config.get("refinement_protocol_file", "configs/refinement_protocol.txt"))
        if not protocol_path.exists():
            logger.warning("Refinement protocol file not found. Using fallback instructions.")
            system_instructions = "You are a professional translator. Refine the translation using the provided sources."
        else:
            system_instructions = protocol_path.read_text(encoding="utf-8")

        final_blocks = []
        step = self.config.get("chunk_size", 10)
        total_blocks = len(s1_blocks)

        logger.info(f"Arbitrating {total_blocks} blocks using windows of {step}.")

        for i in range(0, total_blocks, step):
            s1_win = s1_blocks[i : i + step]
            
            # Sync L1 and Mt based on timestamps of the current S1 window
            t_start = SRTHandler.timestamp_to_seconds(s1_win[0]['start'])
            t_end = SRTHandler.timestamp_to_seconds(s1_win[-1]['end'])

            l1_win = SRTHandler.get_blocks_in_range(l1_blocks, t_start, t_end)
            mt_win = SRTHandler.get_blocks_in_range(mt_blocks, t_start, t_end)

            # Prepare the user prompt (Data context)
            user_prompt = self.prepare_triple_prompt_body(s1_win, l1_win, mt_win)
            
            logger.info(f"Processing window: {s1_win[0]['start']} -> {s1_win[-1]['end']} ({i//step + 1}/{(total_blocks//step)+1})")
            
            try:
                # Call provider with both arguments: system and prompt
                response = self.bot.ask(system_instructions, user_prompt)
                
                refined_blocks = SRTHandler.parse_to_blocks(response)
                
                # Check for block count mismatch
                if len(refined_blocks) != len(s1_win):
                    logger.warning(f"Arbitration mismatch: LLM returned {len(refined_blocks)} blocks, expected {len(s1_win)}.")
                
                final_blocks.extend(refined_blocks)
            except Exception as e:
                logger.error(f"Error during arbitration at window {i}: {e}")
                # Fallback to S1 content to avoid losing the block entirely
                final_blocks.extend(s1_win)

        return SRTHandler.render_blocks(final_blocks)

    def prepare_triple_prompt_body(self, s1: List[Dict], l1: List[Dict], mt: List[Dict]) -> str:
        """Formats the data sources into a clear XML-like structure for the LLM."""
        return f"""
<S1_SOURCE_ORIGINAL>
{SRTHandler.render_blocks(s1)}
</S1_SOURCE_ORIGINAL>

<L1_LITERAL_REFERENCE>
{SRTHandler.render_blocks(l1)}
</L1_LITERAL_REFERENCE>

<MT_DRAFT_STYLE>
{SRTHandler.render_blocks(mt)}
</MT_DRAFT_STYLE>

Please provide the final refined SRT blocks below:"""