import time
import openai
from utu.utils import EnvUtils
import asyncio
from typing import List, Any, Tuple, Optional
from tqdm import tqdm


class LLM:
    def __init__(self):
        EnvUtils.assert_env(["UTU_LLM_TYPE", "UTU_LLM_MODEL", "UTU_LLM_BASE_URL", "UTU_LLM_API_KEY"])
        self.model_name = EnvUtils.get_env("UTU_LLM_MODEL")
        self.embedding_model_name = EnvUtils.get_env("UTU_EMBEDDING_MODEL", "text-embedding-3-small")
        self.client = openai.OpenAI(
            api_key=EnvUtils.get_env("UTU_LLM_API_KEY"),
            base_url=EnvUtils.get_env("UTU_LLM_BASE_URL"),
        )
        self.async_client = openai.AsyncOpenAI(
            api_key=EnvUtils.get_env("UTU_LLM_API_KEY"),
            base_url=EnvUtils.get_env("UTU_LLM_BASE_URL"),
        )
        self.embedding_client = openai.OpenAI(
            api_key="sk-Jp2KWjbRTJniXopDRSkUjTWGyCRUDF15aSvhDbRnrdACMS9C",
            base_url="https://api.nuwaapi.com/v1",
        )

    def chat(self, messages_or_prompt, max_tokens=8192, temperature=0, max_retries=3, return_reasoning=False):
        retries = 0
        while retries < max_retries:
            try:
                if isinstance(messages_or_prompt, str):
                    messages = [{"role": "user", "content": messages_or_prompt}]
                elif isinstance(messages_or_prompt, list):
                    messages = messages_or_prompt
                else:
                    raise ValueError("messages_or_prompt must be a string or a list of messages.")

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1,
                    # presence_penalty=0,
                    # frequency_penalty=0,
                    seed=42,
                    timeout=60.0,
                )
                response_text = response.choices[0].message.content.strip()

                if return_reasoning:
                    reasoning = response.choices[0].message.reasoning_content
                    return response_text, reasoning
                return response_text

            except Exception as e:
                retries += 1
                print(f"API调用失败 (尝试 {retries}/{max_retries}): {type(e).__name__}: {str(e)}")
                if retries >= max_retries:
                    print(f"API调用在 {max_retries} 次重试后最终失败")
                    return ""  # Return empty string instead of None
                time.sleep(3)
        return ""  # Return empty string instead of None
        
    async def chat_async(self, messages_or_prompt, max_tokens=8192, temperature=0, max_retries=3, return_reasoning=False):
        retries = 0
        while retries < max_retries:
            try:
                if isinstance(messages_or_prompt, str):
                    messages = [{"role": "user", "content": messages_or_prompt}]
                elif isinstance(messages_or_prompt, list):
                    messages = messages_or_prompt
                else:
                    raise ValueError("messages_or_prompt must be a string or a list of messages.")

                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1,
                    # presence_penalty=0,
                    # frequency_penalty=0,
                    seed=42,
                    timeout=100.0,
                )
                response_text = response.choices[0].message.content.strip()

                if return_reasoning:
                    reasoning = response.choices[0].message.reasoning_content
                    return response_text, reasoning
                return response_text

            except Exception as e:
                retries += 1
                print(f"API调用失败 (尝试 {retries}/{max_retries}): {type(e).__name__}: {str(e)}")
                if retries >= max_retries:
                    print(f"API调用在 {max_retries} 次重试后最终失败")
                    return ""  # Return empty string instead of None
                await asyncio.sleep(3)
        return ""  # Return empty string instead of None

    async def chat_batch_async(
        self,
        prompts: List[Any],
        concurrency: int,
        max_tokens: int = 8192,
        temperature: float = 0,
        max_retries: int = 3,
        return_reasoning: bool = False,
    ) -> List[Optional[str]]:
        """
        Processes a batch of prompts concurrently using asyncio.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _call_api_with_semaphore(
            prompt_with_id: Tuple[int, Any]
        ) -> Tuple[int, Optional[str]]:
            request_id, prompt = prompt_with_id
            async with semaphore:
                # Using chat_async to leverage its retry logic
                response = await self.chat_async(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_retries=max_retries,
                    return_reasoning=return_reasoning,
                )
                return request_id, response

        tasks = [
            asyncio.create_task(_call_api_with_semaphore((i, p)))
            for i, p in enumerate(prompts)
        ]

        results = [None] * len(prompts)
        # Using a generic description for the progress bar
        pbar = tqdm(total=len(tasks), desc="Processing batch LLM requests")
        for fut in asyncio.as_completed(tasks):
            idx, content = await fut
            results[idx] = content
            pbar.update(1)
        pbar.close()

        return results

    def get_embeddings(self, texts: List[str], batch_size: int = 128, max_retries: int = 3) -> List[Optional[List[float]]]:
        """Gets embeddings for a list of texts."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
            batch_texts = texts[i:i + batch_size]
            retries = 0
            while retries < max_retries:
                try:
                    response = self.embedding_client.embeddings.create(input=batch_texts, model=self.embedding_model_name)
                    embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(embeddings)
                    break # success
                except Exception as e:
                    retries += 1
                    print(f"Error getting embeddings for batch {i}-{i+batch_size} (attempt {retries}/{max_retries}): {e}")
                    if retries >= max_retries:
                        print(f"Failed to get embeddings for batch after {max_retries} retries. Appending None.")
                        all_embeddings.extend([None] * len(batch_texts))
                    time.sleep(3)
        return all_embeddings
                    