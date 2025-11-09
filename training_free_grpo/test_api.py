import asyncio
import time
from openai import OpenAI, AsyncOpenAI

WEBSHOP_PREDICT_ANSWER_REWRITE_TEMPLATE = """
You are an assistant that reads a webpage observation and rewrites it into a **page-type description** for a single user action it took before.

Your goal is to:
- Abstract away IDs, exact numbers, and irrelevant noise.
- Identify what type of page the user land on **after this action**, describing its structure, function, and task-relevant content (e.g., product-details page, search-results page, checkout confirmation page).
- **IMPORTANT** Base the description strictly on current_observation content. Do not infer or add anything not explicitly shown.
- Wrap the final rewritten result in a single pair of <obs> ... </obs> tags.

Here is an example:
The objective of the agent is to: Find me machine wash men's pants with relaxed fit with color: grey, and size: 34w x 32l, and price lower than 70.00 dollars.
Prior to this step, the agent have already taken 2 step(s): [Action 1: 'search[machine wash men's pants with relaxed fit, grey color, size 34w x 32l, price less than $70]'] [Action 2: 'click[b07gyww3ny]'].
The agent is now at step 3 and its current observation is: 'Back to Search' [SEP] '< Prev' [SEP] 'size' [SEP] '32w x 30l' [SEP] '32w x 32l' [SEP] '34w x 30l' [SEP] '34w x 32l' [SEP] '36w x 30l' [SEP] '36w x 32l' [SEP] 'color' [SEP] 'charcoal dust' [SEP] 'coal grey' [SEP] 'navy' [SEP] 'overcast blue' [SEP] 'khaki' [SEP] 'J. Crew - Men's - Sutton Straight-Fit Flex Chino (Multiple Size/Color Options)' [SEP] 'Price: $59.5' [SEP] 'Rating: N.A.' [SEP] 'Description' [SEP] 'Features' [SEP] 'Reviews' [SEP] 'Buy Now'.
Present the rewritten current observation within <obs> </obs> tags.
Expected output: 
<obs>
After clicking on a product ID, this page is a product-details page for men's pants. The page displays various product attributes such as selectable sizes and colors, along with sections for product description, features, reviews, and a "Buy Now" button. Navigation controls include a "Back to Search" link and a "< Prev" button to return to the previous page.
</obs>

Additional example:
User input:  
The objective of the agent is to: Find me machine wash men's dress shirts with cotton spandex, classic fit, short sleeve with color: monaco blue, and size: medium, and price lower than 70.00 dollars.  
Prior to this step, the agent have already taken 1 step(s):[Action 1: 'click[search]'].  
The agent is now at step 2 and its current observation is: 'Search'.  
Present the rewritten the description strictly on current_observation content within <obs> </obs> tags.
Expected output:  
<obs>  
After clicking the search option, this page is a search-input page that contains a search field or prompt and does not display product listings or filters.  
</obs>

Now follow the same rules for the new input:
The objective of the agent is to: {task_description}.
Prior to this step, the agent have already taken 1 step(s):{action_history}.
The agent is now at step 0 and its current observation is: {current_observation}.
Present the rewritten the description strictly on current_observation content within <obs> </obs> tags.
"""


# api_key =  "sk-ZED3BF5uZzwlVhBcCv9GIizknpowIT7iqrVzo3zlTLIqszZ4"      #"sk-zTfWtVEr64wCQ7FisN6n1DetLL7VPkCCuu4a9SnCH7SRfAmZ"         # "sk-a3e4e126622343069e33c404db58fe73"    #"sk-y1XWBmvdmykprTzZ1sOoTAVuO4Vj7CvMcvkVOkJaaGHBKtoI"
# base_url = "https://www.chataiapi.com/v1"              # "https://tb.api.mkeai.com/v1" #"https://dashscope.aliyuncs.com/compatible-mode/v1" #"https://api.nuwaapi.com/v1"
# model_name = "deepseek-chat"  #  "qwen-flash" #"gpt-4o-mini"  #"deepseek-v3.2-exp-fast"


api_key =  "sk-0e06fe2724e14e41b864fbf111ce507d"         # "sk-a3e4e126622343069e33c404db58fe73"    #"sk-y1XWBmvdmykprTzZ1sOoTAVuO4Vj7CvMcvkVOkJaaGHBKtoI"
base_url = "https://api.deepseek.com/v1" #"https://dashscope.aliyuncs.com/compatible-mode/v1" #"https://api.nuwaapi.com/v1"
model_name = "deepseek-chat"

# api_key =  "sk-zTfWtVEr64wCQ7FisN6n1DetLL7VPkCCuu4a9SnCH7SRfAmZ"         # "sk-a3e4e126622343069e33c404db58fe73"    #"sk-y1XWBmvdmykprTzZ1sOoTAVuO4Vj7CvMcvkVOkJaaGHBKtoI"
# base_url = "https://tb.api.mkeai.com/v1" #"https://dashscope.aliyuncs.com/compatible-mode/v1" #"https://api.nuwaapi.com/v1"
# model_name = "deepseek-v3.2-exp-fast"  #  "qwen-flash" #"gpt-4o-mini"  #"deepseek-v3.2-exp-fast"


# api_key =  "sk-4HNfXlNVWSsxpO9bd3vz4a7Z5HILwkGqMR6dJjbg8J4FHhf6"         # "sk-a3e4e126622343069e33c404db58fe73"    #"sk-y1XWBmvdmykprTzZ1sOoTAVuO4Vj7CvMcvkVOkJaaGHBKtoI"
# base_url = "https://api.chatanywhere.org/v1" #"https://dashscope.aliyuncs.com/compatible-mode/v1" #"https://api.nuwaapi.com/v1"
# model_name = "deepseek-v3.2-exp"  #  "qwen-flash" #"gpt-4o-mini"  #"deepseek-v3.2-exp-fast"

# UTU_LLM_TYPE=chat.completions
# UTU_LLM_MODEL="deepseek-chat"    #"deepseek-v3.2-exp"
# UTU_LLM_BASE_URL=https://tb.api.mkeai.com/v1
# UTU_LLM_API_KEY="sk-zTfWtVEr64wCQ7FisN6n1DetLL7VPkCCuu4a9SnCH7SRfAmZ"

# api_key ="sk-BJ0Q7rq6syChNtDqcJiaRNAGMZ0tfXL7reYh3Ucyz37UKUd2" #"sk-y1XWBmvdmykprTzZ1sOoTAVuO4Vj7CvMcvkVOkJaaGHBKtoI"          # "sk-a3e4e126622343069e33c404db58fe73"    #"sk-y1XWBmvdmykprTzZ1sOoTAVuO4Vj7CvMcvkVOkJaaGHBKtoI"
# base_url = "https://api.nuwaapi.com/v1"             #"https://dashscope.aliyuncs.com/compatible-mode/v1" #"https://api.nuwaapi.com/v1"
# model_name = "deepseek-v3.2-exp"  #  "qwen-flash" #"gpt-4o-mini"

concurrency_limit = 100
aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)



async def _call_api_for_rewrite(rewrite_prompt: str, request_id: int, semaphore) -> tuple[int, str]:
    async with semaphore:
        retries = 0
        max_retries = 3
        while retries < max_retries:
            try:
                messages = [
                    {"role": "user", "content": rewrite_prompt}
                ]
                max_tokens =  512
                completion = await aclient.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0,
                    top_p=1,
                    seed=42,
                    # extra_body={"enable_thinking": False}
                )
                content = completion.choices[0].message.content
                return request_id, content
            except Exception as e:
                retries += 1
                print(f"请求 {request_id} 失败 (尝试 {retries}/{max_retries}): {type(e).__name__}: {str(e)}")
                if retries >= max_retries:
                    print(f"请求 {request_id} 在 {max_retries} 次重试后最终失败")
                    return request_id, None
                await asyncio.sleep(0.05)

async def _async_generate_rewrites(rewrite_prompts: list[str]) -> list[str]:
    semaphore = asyncio.Semaphore(concurrency_limit)
    tasks = [
        asyncio.create_task(_call_api_for_rewrite(p, i, semaphore))
        for i, p in enumerate(rewrite_prompts)
    ]
    total = len(tasks)
    results = [None] * total
    for fut in asyncio.as_completed(tasks):
        idx, content = await fut
        results[idx] = content
    return results
    

if __name__ == "__main__":

    rewrite_prompts = []
    for i in range(128):
        task_description='Find me home office furniture sets for dining room, living room with color: taupe | orange, and item shape: runner, and size: 7 ft 9 in x 10 ft 8 in, and price lower than 70.00 dollars.'
        current_observation=''' 'Back to Search' [SEP] 'Page 3 (Total results: 50)' [SEP] '< Prev' [SEP] 'Next >' [SEP] 'B09Q2ZBB1G' [SEP] 'CubiCubi Computer Desk with Shelves, Office Desk with Drawers, 47 Inch Writing Desk with Storage Study Table for Home Office, Living Room, Bedroom, Rustic Brown' [SEP] '$100.0' [SEP] 'B08QVDNJC7' [SEP] 'Valentine‘s Day Love Blue Throw Pillow Covers 18x18 for Home Decor- Love Heart Flower - Modern Linen Cushion Cover Square Home Pillowcase Love Theme Decorative Pillow Covers for Sofa Bed Chair Car' [SEP] '$7.99' [SEP] 'B09B77K9YQ' [SEP] 'Rhomtree 36” Wood Console Table Sideboard Hall Table with 2 Drawers, 1 Cabinet and 1 Shelf Modern Sofa Table for Hallway, Entryway, Living Room, Kitchen (Green with Brown Top)' [SEP] '$309.9' [SEP] 'B09M6VXD6W' [SEP] 'LJP Nightstand Iron Mesh Nightstand, 3 Layer Portable Narrow Side Table can Bearing 50Kg, Sofa End Table Can Be Used as a Small Bookshelf Bedside Table (Color : Black)' [SEP] '$179.82' [SEP] 'B09N55XTPL' [SEP] 'AC Pacific Modern Staggered 6-Shelf Luke Bookcase, Black' [SEP] '$161.0' [SEP] 'B01MR4Q0WA' [SEP] 'Sauder Palladia File Cabinet, Vintage Oak finish' [SEP] '$225.0' [SEP] 'B07T9LZKM7' [SEP] 'Avigers Luxury Decorative European Throw Pillow Cover 18 x 18 Inch Soft Floral Embroidered Cushion Case with Tassels for Couch Bedroom Car 45 x 45 cm, Beige Gold' [SEP] '$16.99' [SEP] 'B09PGQQQDL' [SEP] 'XLBHLH Black LED Chandelier Circular Dimmable 40W 1 Linear Aluminum Pendant Lighting Hanging Ceiling Light for Contemporary Dining Table Entry Kitchen Island' [SEP] '$297.33' [SEP] 'B08K7LDM7Q' [SEP] '2 Pcs Cowhide Throw Pillow Covers Decorative Pillow Cases Farm Animal Brown Cow Hide Skin Print Pillow Case 18 X 18 Inch Velvet Square Cushion Cover for Sofa Bedroom' [SEP] '$17.99' [SEP] 'B07Q87P8DQ' [SEP] 'Permo Vintage Rustic Industrial 3-Lights Kitchen Island Chandelier Triple 3 Heads Pendant Hanging Ceiling Lighting Fixture with Oval Cone Clear Glass Shade (Antique)' [SEP] '$94.99' '''
        action_history='''[Action 1: 'Search[home office furniture]']'''
        available_actions='''click[back to search]',
'click[< prev]',
'click[next >]',
'click[b09q2zbb1g]',
'click[b08qvdnjc7]',
'click[b09b77k9yq]',
'click[b09m6vxd6w]',
'click[b09n55xtpl]',
'click[b01mr4q0wa]',
'click[b07t9lzkm7]',
'click[b09pgqqqdl]',
'click[b08k7ldm7q]',
'click[b07q87p8dq]',
'''
        rewrite_prompt = WEBSHOP_PREDICT_ANSWER_REWRITE_TEMPLATE.format(
            task_description=task_description,
            current_observation=current_observation,
            action_history=action_history,
            available_actions=available_actions
        )
        rewrite_prompts.append(rewrite_prompt)

    print(f"开始发送 {len(rewrite_prompts)} 个API请求...\n")
    
    # 开始计时
    start_time = time.time()
    rewrites = asyncio.run(_async_generate_rewrites(rewrite_prompts))
    end_time = time.time()
    
    # 计算总耗时
    total_time = end_time - start_time
    
    # 统计成功和失败的请求
    success_count = sum(1 for r in rewrites if r is not None)
    fail_count = sum(1 for r in rewrites if r is None)
    print(f"\n{'='*60}")
    print(f"请求完成统计: 成功 {success_count}/{len(rewrites)}, 失败 {fail_count}/{len(rewrites)}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每个请求耗时: {total_time/len(rewrites):.2f} 秒")
    print(f"{'='*60}\n")
    
    #展示rewrites
    # for i, rewrite in enumerate(rewrites):
    #     if rewrite is not None:
    #         print(f"--- 请求 {i} 的结果 ---")
    #         print(rewrite)
    #         print()