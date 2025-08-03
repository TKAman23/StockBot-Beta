# **StockBot** ***Beta***



---



This is StockBot Beta, a first attempt at making an AI that can predict the markets. It is very rough and is definitely not usable to predict the markets yet, and it is not intended for commercial use.




## Contents


### Data

The data used in this project comes from two sources:

1. [yfinance](https://github.com/ranaroussi/yfinance) to fetch financial data from Yahoo Finance. yfinance is an open-source tool not affiliated with Yahoo Inc. Since it is free and managed by Yahoo!, any large scale download attempts from yfinance will likely be terminated.

2. [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID). This is a large stock news dataset *inteded for research purposes only* containing URLs for tickers on the NYSE. To use it requires a scraper. 

> Citations:
>
>> @misc{dong2024fnspid,
>>     title={FNSPID: A Comprehensive Financial News Dataset in Time Series},
>>     author={Zihan Dong and Xinyu Fan and Zhiyuan Peng},
>>     year={2024},
>>     eprint={2402.06698},
>>     archivePrefix={arXiv},
>>     primaryClass={q-fin.ST}
>> }
>
>> @misc{yfinance,
>>     title = {yfinance: Yahoo! Finance market data downloader},
>>     author = {Ran Aroussi},
>>     year = {2018},
>>     publisher = {GitHub},
>>     journal = {GitHub repository},
>>     howpublished = {url{https://github.com/ranaroussi/yfinance}}
>> }


### Data Extraction

For stock price data extraction, please see `get_stock_data_trunc.ipynb`,`get_stock_data.ipynb`,`stock_info_scan.ipynb`.

For stock news data extraction (scrapers, etc.), please see the FNSPID GitHub.

 - Article summary extraction is available at `single_stock/code/single_stock_summaries.ipynb`.

Single stock data extraction is specifically at `single_stock/code`.


### Data Processing

Indicators are calculated via the files `indicator.py` and `tech_indicators.py`.

All other Data Processing occurs right before algorithmic training.


### Model Creation

The model training notebooks are in the `algo_training` folder.

Models will be exported to the `models` folder.




## License


This repository is available under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) license




## Contributions


### Notable Contributors

[Trud12](https://github.com/Trud12)

