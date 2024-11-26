import openai

def gpt4o_generate(prompt, model="gpt-3.5-turbo", max_tokens=500):
    """
    Utility function to generate a response from GPT-3.5.

    Args:
        prompt (str): The prompt to send to GPT-3.5.
        model (str): The GPT model to use (default is "gpt-3.5-turbo").
        max_tokens (int): The maximum number of tokens in the response.

    Returns:
        str: The response from GPT-3.5.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except openai.error.OpenAIError as e:
        # Handle exceptions related to the OpenAI API
        print(f"OpenAI API error: {e}")
        return ""
    except Exception as e:
        # Handle any other exceptions
        print(f"Unexpected error: {e}")
        return ""
