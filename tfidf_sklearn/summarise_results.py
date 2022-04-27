import plotly.express as px
import pandas

df = pandas.read_pickle('cv_results.pkl')
df.drop(['param_processor__hypothesis_processor__stop_words', 'param_processor__hypothesis_processor__use_idf'],
        inplace=True, axis=1)
renaming = {
    'param_processor__premise_processor__stop_words': 'stop_words_removed',
    'param_processor__premise_processor__use_idf': 'IDF Used',
    'param_lr__penalty': 'Regularise',
    'param_lr__solver': 'Optimiser',
    'param_lr__C': 'Regularisation Weight',
}
df.rename(columns=renaming, inplace=True)
df.rename(columns={'mean_test_score': 'Mean Acc (%)'}, inplace=True)
df.replace({
    'Optimiser': {'lbfgs': 'L-BFGS', 'saga': 'SAGA'},
    'Regularise': {'l2': 'L2', 'none': 'No'},
}, inplace=True)

df['Stop Words'] = df['stop_words_removed'].apply(lambda x: 'Removed' if x else 'Present')

print((df.groupby(['Stop Words', 'IDF Used'])['Mean Acc (%)'].mean().round(4) * 100).reset_index().to_latex())
print((df.groupby(['Regularise', 'Optimiser'])['Mean Acc (%)'].mean().round(4) * 100).reset_index().to_latex())

regularised = df[df['Regularisation Weight'] != 1].groupby(['Optimiser', 'Regularisation Weight'])['Mean Acc (%)'].mean().round(4).reset_index()

fig = px.line(regularised, x='Regularisation Weight', y='Mean Acc (%)', title='Regularisation Weight vs. Accuracy (%)', color='Optimiser')
fig.update_layout(legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.90), title_x=0.5)
fig.show()
