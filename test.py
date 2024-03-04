import pandas as pd

test = pd.read_csv('origin_data/CS-Sensors/test.csv', header=None, names=['Timestamp', 'Value'])
value = test['Value'].to_list()

from delta import delta_operator

delta = delta_operator(value)
test['Value'] = delta
test.to_csv('test_delta.csv', index=False, header=None)

from delta import delta_decode

decode = delta_decode(delta)
test['Value'] = decode
test.to_csv('test_delta_decode.csv', index=False, header=None)

from substract_min import substract_min_operator

substract = substract_min_operator(value)
test['Value'] = substract
test.to_csv('test_substract.csv', index=False, header=None)

test = pd.read_csv('origin_data/Cyber-Vehicle/syndata_vehicle1.csv', header=None, names=['Timestamp', 'Value'])
value = test['Value'].to_list()

from xor_float_operator import xor_float_operator

xor = xor_float_operator(value)
test['Value'] = xor
test.to_csv('syndata_vehicle1_xor.csv', index=False, header=None)

from xor_float_decode import xor_float_decode

decode = xor_float_decode(xor)
test['Value'] = decode
test.to_csv('syndata_vehicle1_xor_decode.csv', index=False, header=None)
