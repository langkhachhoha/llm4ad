import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.llm.llm_api_gemini import HttpsApiGemini
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.eoh import EoH

def main():

    # llm = HttpsApi(host='api.openai.com',  # your host endpoint, e.g., api.openai.com/v1/completions, api.deepseek.com
    #                key='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',  # your key, e.g., sk-abcdefghijklmn
    #                model='gpt-3.5-turbo',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
    #                timeout=20
    #                )
    llm = HttpsApiGemini(api_key='AIzaSyCTvxpxSBpdviuLSaQq-mFXiZYA2d-CmME',
                         model='gpt-3.5-turbo'
                         )
    
    task = TSPEvaluation()

    method = EoH(llm=llm,
                 profiler=ProfilerBase(log_dir='logs', log_style='complex'),
                 evaluation=task,
                 max_sample_nums=20,
                 max_generations=5,
                 pop_size=2,
                 num_samplers=1,
                 num_evaluators=1)

    method.run()


if __name__ == '__main__':
    main()
