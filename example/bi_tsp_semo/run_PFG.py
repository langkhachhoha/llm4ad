from llm4ad.task.optimization.bi_tsp_semo import BITSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.llm.llm_api_gemini import HttpsApiGemini 
from llm4ad.tools.llm.llm_api_openai import HttpsApiOpenAI
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.LLMPFG import EoH
from llm4ad.method.LLMPFG import EoHProfiler




def main():

    llm = HttpsApi(host='api.openai.com',  # your host endpoint, e.g., api.openai.com/v1/completions, api.deepseek.com
                   key='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',  # your key, e.g., sk-abcdefghijklmn
                   model='gpt-4o-mini',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=30
                   )
    # llm = HttpsApiGemini(api_key='AIzaSyCTvxpxSBpdviuLSaQq-mFXiZYA2d-CmME',
    #                      model='gemini 2.0 flash',
    #                      )

    # llm = HttpsApiOpenAI(base_url='https://api.openai.com', 
    #                         api_key='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',
    #                         model='gpt-3.5-turbo',
    #                         timeout=30
    #                         )
    task = BITSPEvaluation()

    method = EoH(llm=llm,
                 profiler=EoHProfiler(log_dir='logs', log_style='complex'),
                 evaluation=task,
                 max_sample_nums=100,
                 max_generations=10,
                 pop_size=10,
                 num_samplers=1,
                 num_evaluators=1
                #  llm_review=True
                 )

    method.run()


if __name__ == '__main__':
    main()

