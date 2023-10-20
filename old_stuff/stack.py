import pandas as pd

lgb_single = pd.read_csv('1.csv')
catboost = pd.read_csv('2.csv')
print("Finished Loading the prediction results.")

weight = 0.5
stack = pd.DataFrame()
stack['ParcelId'] = lgb_single['ParcelId']
for col in ['201610', '201611', '201612', '201710', '201711', '201712']:
    stack[col] = weight * catboost[col] + (1 - weight) * lgb_single[col]

print(stack.head())
stack.to_csv('final_stack_3.csv', index=False)