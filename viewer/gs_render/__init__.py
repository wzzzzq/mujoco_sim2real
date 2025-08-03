import os

MOBILE_AI_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if os.getenv('MOBILE_AI_ASSERT_DIR'):
    MOBILE_AI_ASSERT_DIR = os.getenv('MOBILE_AI_ASSERT_DIR')
    print(f'>>> get env "MOBILE_AI_ASSERT_DIR": {MOBILE_AI_ASSERT_DIR}')
else:
    MOBILE_AI_ASSERT_DIR = os.path.join(MOBILE_AI_ROOT_DIR, 'models')