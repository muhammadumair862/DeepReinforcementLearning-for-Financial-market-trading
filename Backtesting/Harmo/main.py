import gc

from utils import *


def logic(df, first_col, second_col, third_col, forth_col):
    entries = (df[first_col] <= df[second_col]) & (df[second_col] <= df[third_col]) & (df[third_col] <= df[forth_col])
    return entries

chunk_size = 1500
n_cores = 8
output_dir = "/home/u0/project0/out"
create_output_folder(output_dir)

file_path = "files/DATA/500T data in order BB and POC at 3 - 6 - 10 - 20 - 30 - 50 - 100 + macd of DELTA.txt"
df = read_data(file_path)
close_price = df['Close']
pf_outputs = [
        # 'start_value',
        # 'min_value',  
        # 'max_value',   
        # 'end_value',
        'total_return', 
        # 'total_trades', 
        # 'total_fees_paid',
        # 'win_rate', 
        # 'best_trade',
        # 'worst_trade',
        # 'avg_winning_trade',
        # 'avg_losing_trade',
        # 'avg_winning_trade_duration', 
		# 'avg_losing_trade_duration'
    ]


start = time.time()


ind = vbt.IndicatorFactory(
    class_name = "compare_indicators",
    short_name = "comp_ind",
    param_names = ['df', 'first_col', 'second_col', 'third_col', 'forth_col'],
    output_names = ["entries"],
        ).with_apply_func(
            logic, 
)

col_names = ['Middle Band', 'Middle Band.1', 'Middle Band.2', 'Middle Band.3',
            'Point of Control', 'Point of Control.1', 'Point of Control.2', 'Point of Control.3',
            'MACD']
cols = [[df] * len(col_names)] + [col_names] * 4


def func(chunk):
    res = ind.run(
        df = chunk[0],
        first_col = chunk[1],
        second_col = chunk[2],
        third_col = chunk[3],
        forth_col = chunk[4]
    )
    entries = res.entries
    entries, exits, _, _ = fixed_sl_exit(df=chunk[0][0], entries=entries, RR=2, SL=3)
    time.sleep(3)
    pf = vbt.Portfolio.from_signals(
        close_price,
        entries=entries,
        exits=exits,
        fees=0.75,
    )
    save_path = os.path.join(output_dir, str(datetime.now()) + '.parquet')
    pf.stats(pf_outputs, agg_func=None).to_parquet(save_path)
    gc.collect()
    print("A chunk has just finished.")
    

parallel_processing(func, cols, chunk_size, n_cores)
print(f"finished in {time.time() - start} seconds")


# hdf_files = glob.glob('files/results/*')
# df_list = []
# for hdf_file in hdf_files:
#     df = pd.read_hdf(hdf_file, key='df')
#     df_list.append(df)

# df = pd.concat(df_list)
# df = df.sort_values(ascending=False)
# print(len(df))
# print(df.head(50))