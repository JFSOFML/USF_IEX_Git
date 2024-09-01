# Base class for vehicles
class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand  # Initialize the brand attribute
        self.model = model  # Initialize the model attribute

    def drive(self):  # Define the drive method
        print("The vehicle is moving forward.")

    def stop(self):  # Define the stop method
        print("The vehicle has stopped.")


# Motorcycle class inherits from Vehicle
class Motorcycle(Vehicle):
    pass  # No additional attributes or methods, just inherits from Vehicle


# Creating an instance of Motorcycle
bike = Motorcycle("Harley-Davidson", "Street 750")
bike.drive()  # Calling the drive method from the Vehicle class
