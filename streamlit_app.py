import streamlit as st
import pandas as pd
from io import StringIO
from pypfopt.efficient_frontier import EfficientFrontier

st.subheader("Input")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  risk_contribution_asset_class_df = pd.read_excel(uploaded_file,sheet_name = 'Risk Contribution - Asset Class').dropna()
  risk_contribution_asset_class_df['Asset'] = [ele['Asset'] + ('' if ele['FX Hedged'] == 'No' else ' (Hedged)') for _,ele in risk_contribution_asset_class_df.iterrows()]
  risk_contribution_asset_class_df = risk_contribution_asset_class_df.set_index('Asset')
  risk_asset_class_corr_mtx_df = pd.read_excel(uploaded_file,sheet_name = 'Risk - Asset Class Corr Mtx').dropna().set_index('Asset Classes')
    
  vol = risk_contribution_asset_class_df['Asset Volatility']
  corr = risk_asset_class_corr_mtx_df
  S = corr.mul(vol,axis='columns').mul(vol,axis='index')
  
  target_return = st.number_input('Insert Target Annual Return (%)',value=1.0) / 100.0
  max_risk_weight = st.number_input('Insert Max. Risk Weight (%)',value=999.0) / 100.0
  input_format_df = pd.DataFrame({'Asset':vol.index})
  input_format_df['Expected Annual Return (%)'] = 0.0
  input_format_df['Lower Bound (%)'] = 0.0
  input_format_df['Upper Bound (%)'] = 100.0
  input_format_df['Risk Weight (%)'] = 0.0
  
  input_txt = st.text_area('Insert optimization inputs',value=input_format_df.to_csv(sep='\t',index=False),height=100)
  input_df = pd.read_csv(StringIO(input_txt),sep='\t').set_index('Asset')
  
  mu = input_df['Expected Annual Return (%)'] / 100.0
  risk_weight = input_df['Risk Weight (%)'] / 100.0

  missing_asset = set(input_df.index) - set(S.index)
  if len(missing_asset) > 0:
    st.error('Missing: ' + ', '.join(missing_asset))
  
  ef = EfficientFrontier(mu, S.loc[mu.index,mu.index], weight_bounds=[tuple(x/100.0) for x in input_df[['Lower Bound (%)','Upper Bound (%)']].values])
  ef.add_constraint(lambda w: risk_weight.values @ w <= max_risk_weight)
  ef.efficient_return(target_return)
  
  weights = ef.clean_weights()
  weights_df = pd.DataFrame.from_dict(weights, orient = 'index')*100.0
  weights_df.columns = ['Optimal Weight (%)']

  expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)
  portfolio_performance_df = pd.DataFrame([expected_annual_return*100.0,annual_volatility*100.0,float(risk_weight @ weights_df)],
                                       index=['Expected Annual Return (%)', 'Annual Volatility (%)','Risk Weight (%)'],
                                       columns=['Portfolio Performance'])
  
  st.subheader("Optimization Result")
  st.dataframe(portfolio_performance_df.style.format(precision=1))
  st.dataframe(weights_df.reindex(input_df.index).style.format(precision=1))
