import dotenv
from langchain_openai import OpenAI, ChatOpenAI
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
import json
import re

# Load OpenAi Key
dotenv.load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
default_model = "gpt-3.5-turbo-16k"


# Prompt Roles:
# User: Initiates the conversation, asking questions.
# System: Sets the initial context.
# Assistant: Generates responses based on user requests and system context.

def sample_requests(with_system_prompt=False):
    sample_prompt = {"role": "user", "content": "what is the first verse of the german anthem?"}
    llm = ChatOpenAI(
        model=default_model,
        max_tokens=1000
    )
    messages = [sample_prompt]
    if with_system_prompt:
        system_prompt = {"role": "system", "content": "Translate everything to english!"}
        messages = [system_prompt, sample_prompt]
    response = llm.invoke(messages)
    print(response)


def generate_executable_python_code(i_user_prompt, i_system_prompt):
    system_prompt = {"role": "system", "content": i_system_prompt}
    user_prompt = {"role": "user", "content": i_user_prompt}
    llm = ChatOpenAI(
        model=default_model,
        max_tokens=500
    )
    messages = [system_prompt, user_prompt]
    response = llm.invoke(messages)
    pattern = r"(?s)```(.*?)```"
    return re.findall(pattern, response.content)[0].replace("python", "")


if __name__ == '__main__':
    step = 4
    if step == 0:
        print("Without system prompt:")
        sample_requests()
        print("With system prompt:")
        sample_requests(with_system_prompt=True)
    elif step == 1:
        print("Generate code and print:")
        # generate_executable_code(False)
        print("Generate code and execute:")
        code = generate_executable_python_code("Generate a function which, for a number x, computes the faculty of x."
                                               " Call the generated function always 'generated_function'",
                                               "You are a python code generator. Generate python code in triple"
                                               " backticks (```).")
        exec(code, globals())
        exec("print(generated_function(5))", globals())
    elif step == 2:
        # data visualisation example
        sales_dataset = pd.read_csv("resources/product_sales_dataset.csv")
        # 1. We want to create a new column "month" which is extracted from the date
        system_prompt = "You are a python code generator. Generate python code in triple backticks (```)."
        user_prompt = ("Return a python method called 'get_month'. This method gets a pandas series "
                       "containing dates as strings, and returns a panda series containing only the month name.")
        code = generate_executable_python_code(user_prompt, system_prompt)
        exec(code, globals())
        exec("sales_dataset['month_name'] = get_month(sales_dataset.Date)", globals())
        print(sales_dataset.head())
    elif step == 3:
        # 2. We want to calculate the profit per product
        # data visualisation example
        sales_dataset = pd.read_csv("resources/product_sales_dataset.csv")
        system_prompt = "You are a python code generator. Generate python code in triple backticks (```)."
        user_prompt = ("Return a python method called 'get_profit_per_day_and_product'. This method gets a pandas "
                       "dataframe containing the columns 'Date', 'Product_Name', 'Product_Cost' and 'Product_Price.' and"
                       " 'Items_Sold. Compute per product name and date the respective profit by computing the difference"
                       " between  the price and the costs times the items sold on this date."
                       " This shall be done for each day and returned as a new series.")
        code = generate_executable_python_code(user_prompt, system_prompt)
        exec(code, globals())
        exec("sales_dataset = get_profit_per_day_and_product(sales_dataset)", globals())
        print(sales_dataset)
    elif step == 4:
        # 3. We want to plot the average product profit per day.
        sales_dataset = pd.read_csv('resources/product_sales_dataset.csv')
        system_prompt = "You are a python code generator. Generate python code in triple backticks (```)."
        user_prompt = ("Generate a method called plot_average_product_profit_per_day.This method gets a pandas "
                       "dataframe containing the columns 'Date', 'Product_Name', 'Product_Cost' and 'Product_Price.' and"
                       " 'Items_Sold. It first computes per day for each product the profit by computing the difference"
                       " between the price and the costs times the items sold on this date. Then it combines the profits"
                       " from all products on this date."
                       "Finally, it visualizes each day as chart where x is the day and y is the average profit.")
        code = generate_executable_python_code(user_prompt, system_prompt)
        exec(code, globals())
        exec("plot_average_product_profit_per_day(sales_dataset)", globals())
