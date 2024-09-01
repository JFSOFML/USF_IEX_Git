###Dictionaries
# Construction of Dictionaries
empty_dict = {}
vehicle = {
    "Brand": "Ford",
    "Year": 2020,
    "Model": "Mustang",
}  ##all keys above need a "," except the last one. Also cannot have more then 1 value unless you put it in a dicitonary.
print(vehicle)
vehicle["Brand"]

assesment = {}

scores = [("annie", 10), ("Maddie", 5), ("Becky", 3)]

for name, score in scores:

    assesment[name] = score

print(assesment)


# Ordered / unordered
# ordered: order of the keys and values are maintainted
# undered: order of the keys and values are not maintainted

assesment.keys()
