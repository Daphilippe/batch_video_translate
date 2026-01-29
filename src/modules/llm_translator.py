import logging
import time
from pathlib import Path
from typing import List, Dict
from modules.translator import BaseTranslator
from utils.srt_handler import SRTHandler
from modules.providers.base_provider import LLMProvider

logger = logging.getLogger(__name__)

class LLMTranslator(BaseTranslator):
    def __init__(self, input_dir, output_dir, provider: LLMProvider, config: Dict):
        super().__init__(input_dir, output_dir, bot=provider, extensions=(".srt",))
        self.provider = provider
        self.config = config
        self.chunk_size = config.get("chunk_size", 10)
        self.system_instructions = self._load_custom_prompt(config)

    def _load_custom_prompt(self, config: Dict) -> str:
        prompt_path = Path(config.get("prompt_file", "configs/system_prompt.txt"))
        if not prompt_path.exists():
            logger.warning(f"Prompt file not found at {prompt_path}. Using fallback.")
            raw_prompt = "Translate from {source_lang} to {target_lang}:"
        else:
            with open(prompt_path, "r", encoding="utf-8") as f:
                raw_prompt = f.read()
        
        return raw_prompt.format(
            source_lang=config.get("source_lang", "English"),
            target_lang=config.get("target_lang", "French")
        )

    def is_time_close(self, t1: str, t2: str, margin: float = 0.150) -> bool:
        try:
            s1 = SRTHandler.timestamp_to_seconds(t1)
            s2 = SRTHandler.timestamp_to_seconds(t2)
            return abs(s1 - s2) <= margin
        except Exception as e:
            logger.debug(f"Timestamp comparison error: {e}")
            return False

    def process_file(self, input_file: Path):
        output_file = self.get_output_path(input_file, ".srt")
        logger.info(f"--- Processing File: {input_file.name} ---")
        
        # 1. Loading Source and existing Draft
        with open(input_file, "r", encoding="utf-8") as f:
            s1_blocks = SRTHandler.parse_to_blocks(f.read())
        logger.info(f"Loaded {len(s1_blocks)} blocks from source (S1).")
        
        mt_blocks_map = {}
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                raw_mt = SRTHandler.parse_to_blocks(f.read())
                mt_blocks_map = {b['start']: b for b in raw_mt}
            logger.info(f"Found existing draft (Mt) with {len(raw_mt)} blocks.")
        else:
            logger.info("No existing draft found. Starting from scratch.")

        # 2. Surgical Reconstruction Logic
        final_blocks = []
        i = 0
        total_s1 = len(s1_blocks)
        mismatch_count = 0

        while i < total_s1:
            s1_blk = s1_blocks[i]
            start_time = s1_blk['start']
            
            # CASE A: Block exists and matches
            if start_time in mt_blocks_map:
                final_blocks.append(mt_blocks_map[start_time])
                i += 1
            # CASE B: Gap detected
            else:
                mismatch_count += 1
                start_gap = i
                logger.info(f"Gap detected at index {i} (Timestamp: {start_time}). Searching for end of gap...")
                
                while i < total_s1 and s1_blocks[i]['start'] not in mt_blocks_map:
                    i += 1
                end_gap = i
                
                gap_len = end_gap - start_gap
                logger.info(f"Gap identified: {gap_len} missing block(s) [Indices {start_gap} to {end_gap-1}].")
                
                gap_to_translate = s1_blocks[start_gap:end_gap]
                
                # Context Management
                context_prev = [s1_blocks[start_gap - 1]] if start_gap > 0 else []
                context_next = [s1_blocks[end_gap]] if end_gap < total_s1 else []
                
                logger.info(f"Requesting LLM translation for gap cluster. Context: {'Yes' if context_prev else 'No'}(prev), {'Yes' if context_next else 'No'}(next).")
                
                translated_gap = self._translate_with_context(context_prev, gap_to_translate, context_next)
                
                final_blocks.extend(translated_gap)
                
                # Update Cache File Immediately
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(SRTHandler.render_blocks(final_blocks))
                logger.debug(f"Partial save completed after gap fix at index {start_gap}.")

        logger.info(f"--- Finished {input_file.name} ---")
        logger.info(f"Total gaps fixed: {mismatch_count}. Final block count: {len(final_blocks)}.")

    def _translate_with_context(self, prev: List[Dict], gap: List[Dict], next_blk: List[Dict]) -> List[Dict]:
        full_context = prev + gap + next_blk
        context_text = SRTHandler.render_blocks(full_context)
        
        prompt = f"""
[SYSTEM ROLE: Professional SRT Translator]
I will provide a sequence of subtitle blocks.

INPUT DATA:
- PREVIOUS CONTEXT (DO NOT RETURN): {len(prev)} blocks
- TARGET TO TRANSLATE (RETURN ONLY THESE): {len(gap)} blocks
- NEXT CONTEXT (DO NOT RETURN): {len(next_blk)} blocks

STRICT RULES:
1. Translate the TARGET blocks from {self.config.get('source_lang')} to {self.config.get('target_lang')}.
2. OUTPUT ONLY the {len(gap)} translated SRT blocks.
3. NO EXPLANATIONS, NO COMMENTS, NO PREAMBLE.
4. If a block is in Russian, it MUST be translated to French.
5. Keep the exact same SRT structure (Index, Timestamp, Text).

BLOCKS TO PROCESS:
{context_text}

FINAL OUTPUT (SRT ONLY):
"""
        try:
            start_time_llm = time.time()
            raw_response = self.provider.ask(self.system_instructions, prompt)
            duration = time.time() - start_time_llm
            
            translated_blocks = SRTHandler.parse_to_blocks(raw_response)
            
            logger.info(f"LLM response received in {duration:.2f}s. Parsed {len(translated_blocks)} blocks.")

            if len(translated_blocks) != len(gap):
                logger.error(f"CRITICAL MISMATCH: LLM returned {len(translated_blocks)} blocks but {len(gap)} were expected.")
                logger.warning("Falling back to source blocks for this gap to maintain synchronization.")
                return gap
                
            return translated_blocks
        except Exception as e:
            logger.error(f"LLM request failed: {str(e)}")
            return gap