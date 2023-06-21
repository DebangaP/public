changepoint_scale = [0.05, 0.15]

for cpscale in changepoint_scale:

    print('Running for Changepoint: ' + str(cpscale))
    df_temp = pd.DataFrame()
    for file in glob.glob(TS_data_Dir + '*.csv'):
        file_name = os.path.basename(file)

        df_price_ts = pd.read_csv(file)

        m = Prophet(changepoint_prior_scale = cpscale)
        m.fit(df_price_ts)

        future = m.make_future_dataframe(periods=365)

        forecast = m.predict(future)
        #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

        forecast.to_csv(forecastDir + file_name.replace(".csv", "") + str(process_date.strftime('%d%m%y')) +  ".csv")

        print(round(forecast['yhat'].iloc[-1], 0))

        df_temp = df_temp.append({'Symbol': file_name.replace(".csv", ""), "1Yr_target": round(forecast['yhat'].iloc[-1], 0)}, ignore_index= True)

    df_temp.to_csv(forecastDir + _run_for + '_' + str(cpscale) + '_' + str(process_date.strftime('%d%m%y')) +  ".csv", columns=['Symbol', '1Yr_target'])


print(" Prophet completed ")
