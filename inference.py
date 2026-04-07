import os
from openai import OpenAI
from envs import email_env, data_clean_env, code_review_env

# Read environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def run_task(env, task_name):
    obs = env.reset()
    print(f"[START] task={task_name} env=openenv model={MODEL_NAME}")
    rewards = []
    step_count = 0
    success = False
    error = "null"

    try:
        while True:
            step_count += 1
            # Example: agent picks an action (stubbed here)
            action = "dummy_action"
            obs, reward, done, info = env.step(action)
            rewards.append(f"{reward:.2f}")
            print(f"[STEP] step={step_count} action={action} reward={reward:.2f} done={str(done).lower()} error={error}")
            if done:
                success = reward > 0.5
                break
    except Exception as e:
        error = str(e)
    finally:
        print(f"[END] success={str(success).lower()} steps={step_count} rewards={','.join(rewards)}")

if __name__ == "__main__":
    run_task(email_env.EmailEnv(), "email-triage")
    run_task(data_clean_env.DataCleanEnv(), "data-cleaning")
    run_task(code_review_env.CodeReviewEnv(), "code-review")