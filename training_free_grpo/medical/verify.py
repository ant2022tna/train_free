from math_verify.errors import TimeoutException
#from math_verify.metric import math_metric
#from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


def eval_for_multiple_choice(input_text: str, final_answer: str, target: str) -> bool:
    """
    Evaluates if the final answer matches the target using pattern matching.
    
    Args:
        input_text (str): The original question text including options
        final_answer (str): The model's answer
        target (str): The correct answer
    
    Returns:
        bool: True if answer is correct, False otherwise
    """
    # Handle empty or None inputs
    if not final_answer or not target:
        return False
    
    def clean_text(text: str) -> str:
        if not text:
            return ""
        return text.lower().strip().replace('`', '').replace('(', '').replace(')', '')
    
    def extract_option_text(input_text: str, option_letter: str) -> str:
        try:
            # Try different formats of options sections
            options_section = ""
            if 'options:' in input_text.lower():
                options_section = input_text.lower().split('options:')[1].strip()
            elif 'choices:' in input_text.lower():
                options_section = input_text.lower().split('choices:')[1].strip()
            
            if not options_section:
                # Try to find options in the format (A) text, (B) text
                lines = input_text.lower().split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith(f'({option_letter})') or line.strip().startswith(f'{option_letter})'):
                        return line.split(')', 1)[1].strip()
                
            # Process the options section if found
            for line in options_section.split('\n'):
                line = line.strip()
                if line.startswith(f'({option_letter})') or line.startswith(f'{option_letter})'):
                    return line.split(')', 1)[1].strip()
                # Handle options like "A. text" format
                if line.startswith(f'{option_letter}.'):
                    return line.split('.', 1)[1].strip()
        except:
            return ''
        return ''

    # Full option match (A), (B), etc. (e.g., (A) == (A))
    if final_answer == target:
        return True

    # Clean and normalize inputs
    clean_answer = clean_text(final_answer)
    clean_target = clean_text(target)
    
    # Handle target formats: (A), A), A, etc.
    target_letter = ""
    if len(clean_target) == 1:
        target_letter = clean_target
    elif clean_target.endswith(')'):
        target_letter = clean_target[-2]
    else:
        # Extract the last character if it's a letter a-d or A-D
        last_char = clean_target[-1]
        if last_char in 'abcd':
            target_letter = last_char
    
    # Direct letter match (a, b, c, d)
    if len(clean_answer) == 1 and clean_answer in 'abcd' and clean_answer == target_letter:
        return True
    
    # Handle answer formats like "A" or "A."
    if clean_answer.startswith(target_letter) and (len(clean_answer) == 1 or 
                                                  (len(clean_answer) == 2 and clean_answer[1] == '.')):
        return True
    
    # Handle answer formats like "Option A" or "Answer is A"
    if clean_answer.endswith(target_letter) and (clean_answer[-2:] == f" {target_letter}" or 
                                               clean_answer[-3:] == f" {target_letter}."):
        return True
    
    # Text content match - check if the target option text is in the answer
    target_text = extract_option_text(input_text, target_letter)
    
    if target_text and target_text in clean_answer:
        return True
    
    # Handle numerical answers (if target is a number and answer contains that number)
    if target_letter.isdigit() and target_letter in clean_answer:
        return True
        
    return False

def classify_answer(input_text: str, final_answer: str) -> str:
    """
    Classifies the final answer to a multiple choice option.

    Args:
        input_text (str): The original question text including options.
        final_answer (str): The model's answer.

    Returns:
        str: The classified option letter (e.g., "A", "B") or an empty string if it cannot be determined.
    """
    if not final_answer:
        return ""

    def clean_text(text: str) -> str:
        if not text:
            return ""
        return text.lower().strip().replace('`', '').replace('(', '').replace(')', '')

    def extract_option_text(input_text: str, option_letter: str) -> str:
        try:
            # Try different formats of options sections
            options_section = ""
            if 'options:' in input_text.lower():
                options_section = input_text.lower().split('options:')[1].strip()
            elif 'choices:' in input_text.lower():
                options_section = input_text.lower().split('choices:')[1].strip()
            
            if not options_section:
                # Try to find options in the format (A) text, (B) text
                lines = input_text.lower().split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith(f'({option_letter})') or line.strip().startswith(f'{option_letter})'):
                        return line.split(')', 1)[1].strip()
                
            # Process the options section if found
            for line in options_section.split('\n'):
                line = line.strip()
                if line.startswith(f'({option_letter})') or line.startswith(f'{option_letter})'):
                    return line.split(')', 1)[1].strip()
                # Handle options like "A. text" format
                if line.startswith(f'{option_letter}.'):
                    return line.split('.', 1)[1].strip()
        except:
            return ''
        return ''

    clean_answer = clean_text(final_answer)
    possible_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # 1. Check for single-letter answers or formats like "A."
    if len(clean_answer) == 1 and clean_answer.upper() in possible_options:
        return clean_answer.upper()
    if len(clean_answer) == 2 and clean_answer.endswith('.') and clean_answer[0].upper() in possible_options:
        return clean_answer[0].upper()

    # 2. Check for formats like "Option A" or "The answer is A"
    words = clean_answer.split()
    if len(words) > 0:
        last_word = words[-1].strip('.').upper()
        if last_word in possible_options:
            return last_word

    # 3. Match against the full text of the option
    for option in possible_options:
        option_text = extract_option_text(input_text, option.lower())
        if option_text and clean_text(option_text) in clean_answer:
            return option

    return ""

def extract_answer(
    response: str,
) -> str:
    """
    Extracts the final answer from the model response.

    Arguments:
        response : str : The response from the model.

    Returns:
        str : The extracted final answer (if not found, returns "No final answer found").
    """
    if "<answer>" in response:
        # <answer> (content) </answer>
        try:
            txt = response.split("<answer>")[-1].strip()
            txt = txt.split("</answer>")[0].strip()
            return txt
        except:
            return "No final answer found"
    else:
        if not("FINAL ANSWER" in response):
            return "No final answer found"
        try:
            response = response.split("FINAL ANSWER")[-1].strip()
            if response[0] == ":":
                response = response[1:].strip()

            # First decide whether to split by "```" or "'''" based on the presence of "```" or "'''"
            idx_1 = response.find("'''")
            idx_2 = response.find("```")
            if min(idx_1, idx_2) != -1: 
                if idx_1 < idx_2:
                    response = response.split("'''")[1].strip()
                else:
                    response = response.split("```")[1].strip()
            else:
                if idx_1 == -1:
                    response = response.split("```")[1].strip()
                else:
                    response = response.split("'''")[1].strip()

            # Special case for P3-Test task: If the first line contains "python" then remove it
            if response.split("\n")[0].strip().lower() == "python":
                response = "\n".join(response.split("\n")[1:]).strip()
            return response
        except:
            return "No final answer found"

def verify_func(sample: dict, ground_truth: str, timeout_score: float = 0) -> float:
    model_output = sample["response"]
    model_input = sample["problem"]

    extracted_answer=extract_answer(model_output)

    try:
        ret_score = eval_for_multiple_choice(model_input, extracted_answer,ground_truth)
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return float(ret_score)