from llm4ad.task.optimization.bi_cvrp import BICVRPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.llm.llm_api_gemini import HttpsApiGemini 
from llm4ad.tools.llm.llm_api_openai import HttpsApiOpenAI
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.LLMPFG import EoH
from llm4ad.method.LLMPFG import EoHProfiler




# Set your LLM API key here
with open("secret.txt", "r") as f:
    secret = f
    llm_api_key = secret.readline().strip()


def main():

    llm = HttpsApi(host='api.openai.com',  # your host endpoint, e.g., api.openai.com/v1/completions, api.deepseek.com
                   key=llm_api_key, 
                   model='gpt-4o-mini',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=30
                   )
    # llm = HttpsApiGemini(api_key='',
    #                      model='gemini 2.0 flash',
    #                      )

    # llm = HttpsApiOpenAI(base_url='https://api.openai.com', 
    #                         api_key='',
    #                         model='gpt-3.5-turbo',
    #                         timeout=30
    #                         )
    task = BICVRPEvaluation()

    method = EoH(llm=llm,
                 profiler=EoHProfiler(log_dir='logs', log_style='complex'),
                 evaluation=task,
                 max_sample_nums=100,
                 max_generations=10,
                 pop_size=2,
                 num_samplers=1,
                 num_evaluators=1,
                 llm_review=True
                 )

    method.run()


if __name__ == '__main__':
    main()

