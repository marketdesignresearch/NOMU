import pickle

with open("/tests/test_runs/2020_09_11_0649_all_test/with_old_DE/UB/optimizer_data_step0.pickle", 'rb') as handle:
    data = pickle.load(handle)
    data["model"] = "UB"
with open(
        "/tests/test_runs/2020_09_11_0649_all_test/with_old_DE/UB/optimizer_data_step0.pickle",
            'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
