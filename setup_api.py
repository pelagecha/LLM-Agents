# import os
# from dotenv import load_dotenv
# import yaml

# # Load the .env file
# load_dotenv()

# # Access the API key from the environment
# api_key = os.getenv('OPENAI_API_KEY')

# # YAML configuration with the API key
# yaml_content = {
#     'module': 'src.client.agents.HTTPAgent',
#     'parameters': {
#         'url': 'https://api.openai.com/v1/chat/completions',
#         'headers': {
#             'Content-Type': 'application/json',
#             'Authorization': f'Bearer {api_key}'
#         },
#         'body': {
#             'temperature': 0
#         },
#         'prompter': {
#             'name': 'role_content_dict',
#             'args': {
#                 'agent_role': 'assistant'
#             }
#         },
#         'return_format': "{response[choices][0][message][content]}"
#     }
# }





# with open('configs/agents/openai-chat.yaml', 'w') as yaml_file:
#     yaml.dump(yaml_content, yaml_file, default_flow_style=False)
#

"""
module: src.client.agents.HTTPAgent
parameters:
    url: https://api.openai.com/v1/chat/completions
    headers:
        Content-Type: application/json
        Authorization: Bearer mykey
    body:
        temperature: 0
    prompter:
        name: role_content_dict
        args:
            agent_role: assistant
    return_format: "{response[choices][0][message][content]}"
"""