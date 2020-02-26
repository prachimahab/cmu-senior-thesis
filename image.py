import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt


c = np.load('vgg19_eval_fc6.npy')
# print(len(c), len(c[0]))
# print(c[0])
# print(c[:][0])



with open('convnet_image_orders_fc6.p', 'rb') as f:
    x = pickle.load(f)
image_orders = x

with open('convnet_image_orders_fc6.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(image_orders)



# Indexes for each category
# objects
bird_indexes = [3319,3621,4532,3897,3002,3611,3905,3662,2160,1055,2156]
train_indexes = [2216,1726,2731,1305,4363,837,2546,2123,2350,1948,4260]
dog_indexes = [4415,4136,3756,3616,4769,4053,3037,4161,4014,3278,4703]
oranges_indexes = [2487,4732,3230,2121,3879,2299,2048,1313,2203,2410,2847]
table_indexes = [3332,817,1118,934,1935,1101,3804,78,2871,694,746]
cat_indexes = [1773,2384,1523,2300,2538,4672,2094,1926,2597,2525,2093]
bear_indexes = [2267,1432,2020,2234,1041,1961,2115,3021,2989,2323,4754]
clocktower_indexes = [2164,2439,2668,1839,2992,1811,1768,2128,2694,2395,1578]
squirrel_indexes = [4100,3136,3069,3705]
insect_indexes = [3099,4573,3638,3916,3379,3526,4062,3394,3966,3419,3646]
donut_indexes =[2791,1749,1488,1673,1392,2162,2334,1758,1261]

# scenes
surfing_indexes = [2191,2453,1275,1011,2858,2083,2575,1398,2822,1399,2008]
motorcycling_indexes = [2132,1317,2917,1657,1672,2406,2547,2591,1688,1540,2903]
beach_indexes = [2230,1129,896,169,2537,1507,2949,2184,1576,2154]
grocerystore_indexes = [4849,969,1265,375,724,2680,1532,363,1962,295,2471]
bus_indexes = [2665,2010,2512,2813,4513,2583,1729,2023,1349,1045,1687]
banquethall_indexes = [594,619,4125,468,801]
parkinggarage_indexes = [989,978,970,651]

# mixed categories
mixed_indexes = [3897,2731,3616,2453,2917,4732,1118,1523,2020,1839,1129]

def image_features(c, category_indexes):
    data = []
    for val in  category_indexes:
        image_data = c[val]
        data.append(image_data)
    return np.array(data)

def image_corr(category, name):
    corr = np.corrcoef(category)
    total = np.triu(corr[0:], 1)
    #because it is symmetric --> and get rid of 1
    total_mean= np.mean(total)
    plt.imshow(corr)
    #plt.colorbar()
    plt.title(name)
    plt.savefig(name + "_image_corr.png")
    print(name,total_mean)
    return total_mean


def imagedat(c, category_indexes):
	pull_data = c

	Bird = image_features(pull_data, bird_indexes)
	Train = image_features(pull_data, train_indexes)
	Dog = image_features(pull_data, dog_indexes)
	Surfing = image_features(pull_data, surfing_indexes)
	Motorcyling = image_features(pull_data, motorcycling_indexes)
	Oranges = image_features(pull_data, oranges_indexes)
	Table = image_features(pull_data, table_indexes)
	Cat = image_features(pull_data, cat_indexes)
	Bear = image_features(pull_data, bear_indexes)
	Clocktower = image_features(pull_data, clocktower_indexes)
	Beach = image_features(pull_data, beach_indexes)
	GroceryStore = image_features(pull_data, grocerystore_indexes)
	Bus = image_features(pull_data, bus_indexes)
	Squirrel = image_features(pull_data, squirrel_indexes)
	Insect = image_features(pull_data, insect_indexes)
	Donut = image_features(pull_data, donut_indexes)
	BanquetHall = image_features(pull_data, banquethall_indexes)
	ParkingGarage = image_features(pull_data, parkinggarage_indexes)
	Mixed = image_features(pull_data, mixed_indexes)

	birds_corr = image_corr(Bird, "Bird")
	train_corr = image_corr(Train, "Train")
	dog_corr = image_corr(Dog, "Dog")
	surfing_corr = image_corr(Surfing, "Surfing")
	motorcycling_corr = image_corr(Motorcyling, "Motorcycling")
	oranges_corr = image_corr(Oranges, "Oranges")
	table_corr = image_corr(Table, "Table")
	cat_corr = image_corr(Cat, "Cat")
	bear_corr = image_corr(Bear, "Bear")
	clocktower_corr = image_corr(Clocktower, "Clocktower")
	beach_corr = image_corr(Beach, "Beach")
	grocerystore_corr = image_corr(GroceryStore, "Grocery Store")
	bus_corr = image_corr(Bus, "Bus")
	squirrel_corr = image_corr(Squirrel, "Squirrel")
	insect_corr = image_corr(Insect, "Insect")
	donut_corr = image_corr(Donut, "Donut")
	banquethall_corr = image_corr(BanquetHall, "Banquet Hall")
	parkinggarage_corr = image_corr(ParkingGarage, "Parking Garage")
	mixed_corr = image_corr(Mixed, "Mixed")
	# plt.bar(["RepMean", "TotalMean", "NonrepMean"], 300, 0.8, birds_corr)
	# plt.show()
	return [birds_corr, train_corr, dog_corr, surfing_corr, motorcycling_corr, oranges_corr, table_corr,
			cat_corr, bear_corr, clocktower_corr, beach_corr, grocerystore_corr, bus_corr, squirrel_corr, insect_corr,
			donut_corr, banquethall_corr, parkinggarage_corr, mixed_corr]

imagedat(c, mixed_indexes)
corr_array = imagedat(c, mixed_indexes)
print(corr_array)

#print(image_features(c, train_indexes))
