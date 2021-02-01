import pandas as pd
import matplotlib.pyplot as plt
import time
from appliance_param import appliance_param


def generate(appliance_name_list, data_dir, aggregate_mean, aggregate_std, save_path):
    start_time = time.time()
    sample_seconds = 60
    validation_percent = 10
    nrows = None
    debug = True

    for appliance_name in appliance_name_list:
        print('\n' + appliance_name)
        train = pd.DataFrame(columns=['aggregate', appliance_name])
        index_count = {}

        for h in appliance_param[appliance_name]['houses']:
            mains_df = generate_mains(data_dir, appliance_name, nrows, sample_seconds, debug, h)
            app_df, index_count = generate_appliance(data_dir, appliance_name, nrows, sample_seconds, debug, h, index_count)
            df_align = generate_mains_appliance(mains_df, app_df, appliance_name, sample_seconds, debug)
            normalization(df_align, appliance_name, aggregate_mean, aggregate_std)

            if h == appliance_param[appliance_name]['test_build']:
                df_align.to_csv(save_path + appliance_name + '_test_.csv', mode='a', index=False, header=False)
                print("    Size of test set is {:.4f} M rows.".format(len(df_align) / 10 ** 6))
                continue

            train = train.append(df_align, ignore_index=True)
            del df_align

        # Validation CSV
        val_len = int((len(train) / 100) * validation_percent)
        val = train.tail(val_len)
        val.reset_index(drop=True, inplace=True)
        train.drop(train.index[-val_len:], inplace=True)
        val.to_csv(save_path + appliance_name + '_validation_' + '.csv', mode='a', index=False, header=False)

        # Training CSV
        train.to_csv(save_path + appliance_name + '_training_.csv', mode='a', index=False, header=False)

        print("    Size of total training set is {:.4f} M rows.".format(len(train) / 10 ** 6))
        print("    Size of total validation set is {:.4f} M rows.".format(len(val) / 10 ** 6))
        del train, val
        print("\nPlease find files in: " + save_path)
        print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))


def generate_mains(data_dir, appliance_name, nrows, sample_seconds, debug, h):
    mains1_df = pd.read_table(data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' + str(1) + '.dat',
                              sep="\s+", nrows=nrows, usecols=[0, 1], names=['time', 'mains1'], dtype={'time': str}, )
    mains2_df = pd.read_table(data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' + str(2) + '.dat',
                              sep="\s+", nrows=nrows, usecols=[0, 1], names=['time', 'mains2'], dtype={'time': str}, )

    mains1_df['time'] = pd.to_datetime(mains1_df['time'], unit='s')
    mains2_df['time'] = pd.to_datetime(mains2_df['time'], unit='s')
    mains1_df.set_index('time', inplace=True)
    mains2_df.set_index('time', inplace=True)
    mains_df = mains1_df.join(mains2_df, how='outer')
    mains_df['aggregate'] = mains_df.iloc[:].sum(axis=1)
    mains_df.reset_index(inplace=True)

    # 此处重采样
    mains_df.set_index('time', inplace=True)
    mains_df = mains_df.resample(str(sample_seconds) + 'S').fillna(method='backfill', limit=1)
    mains_df.reset_index(inplace=True)
    # deleting original separate mains
    del mains_df['mains1'], mains_df['mains2']

    if debug:
        print("    mains_df:")
        print(mains_df.head())
        plt.plot(mains_df['time'], mains_df['aggregate'])
        plt.show()

    del mains1_df, mains2_df
    return mains_df


def generate_appliance(data_dir, appliance_name, nrows, sample_seconds, debug, h, index_count):
    if h not in index_count.keys():
        index_channel = appliance_param[appliance_name]['houses'].index(h)
        index_count[h] = appliance_param[appliance_name]['houses'].index(h)
    else:
        index_channel = appliance_param[appliance_name]['houses'].index(h, index_count[h])

    print('    ' + data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' +
          str(appliance_param[appliance_name]['channels'][index_channel]) + '.dat')

    app_df = pd.read_table(data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' +
                           str(appliance_param[appliance_name]['channels'][index_channel]) + '.dat',
                           sep="\s+", nrows=nrows, usecols=[0, 1], names=['time', appliance_name],
                           dtype={'time': str}, )
    index_count[h] = index_count[h] + 1
    app_df['time'] = pd.to_datetime(app_df['time'], unit='s')
    app_df.set_index('time', inplace=True)
    app_df = app_df.resample(str(sample_seconds) + 'S').fillna(method='backfill', limit=1)
    app_df.reset_index(inplace=True)
    if debug:
        print("app_df:")
        print(app_df.head())
        plt.plot(app_df['time'], app_df[appliance_name])
        plt.show()
    return app_df, index_count


def generate_mains_appliance(mains_df, app_df, appliance_name, sample_seconds, debug):
    mains_df.set_index('time', inplace=True)
    app_df.set_index('time', inplace=True)
    df_align = mains_df.join(app_df, how='outer').resample(str(sample_seconds) + 'S').fillna(method='backfill', limit=1)
    df_align = df_align.dropna()
    df_align.reset_index(inplace=True)
    if debug:
        print("df_align_time:")
        print(df_align.head())
    del mains_df, app_df, df_align['time']
    if debug:
        print("df_align:")
        print(df_align.head())
        plt.plot(df_align['aggregate'].values)
        plt.plot(df_align[appliance_name].values)
        plt.show()
    return df_align


def normalization(df_align, appliance_name, aggregate_mean, aggregate_std):
    # Normalization
    if 'mean' in appliance_param[appliance_name]:
        mean = appliance_param[appliance_name]['mean']
    else:
        mean = appliance_param['default_param']['mean']

    if 'std' in appliance_param[appliance_name]:
        std = appliance_param[appliance_name]['std']
    else:
        std = appliance_param['default_param']['std']

    df_align['aggregate'] = (df_align['aggregate'] - aggregate_mean) / aggregate_std
    df_align[appliance_name] = (df_align[appliance_name] - mean) / std