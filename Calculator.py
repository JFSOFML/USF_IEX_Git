while True:
    print(
        """
    Calculator v1.0 Menu
    1. Add two numbers
    2. Subtract two number
    3. Divde two numbers
    4. Multiple two numbers
    5. Calculate 8% Tax on a value
    6. Calculate monthly payments, given a total amaount and a number of years
    7. Quit


   """
    )

    # print user input
    userchoice = int(input("Please choose an option from 1 - 6:"))

    # if user chooses 1, add 2 numbers
    if userchoice == 1:
        print("ADD TWO NUMBER")
        number_1 = float(input("First number:"))
        number_2 = float(input("Second number:"))
        result = number_1 + number_2
        print("Result:", result)

    # else if
    elif userchoice == 2:
        print("SUBTRACT TWO NUMBERS")
        number_1 = float(input("First number: "))
        number_2 = float(input("Second number: "))
        result = number_1 - number_2
        print("Result:", result)

    # fill in the rest
    elif userchoice == 3:
        print("DIVIDING TWO NUMBERS")
        number_1 = float(input("First number: "))
        number_2 = float(input("Second number: "))
        result = number_1 / number_2
        print("Result:", result)

    elif userchoice == 4:
        print("MULTIPLY TWO NUMBERS")
        number_1 = float(input("First number: "))
        number_2 = float(input("Second number: "))
        result = number_1 * number_2
        print("Result:", result)

    elif userchoice == 5:
        print("CALCULATING 8% TAX ON A VALUE")
        amount = float(input("Enter amount:"))
        result = amount * 0.08
        print("Result:", result)

    elif userchoice == 6:
        print("CALCULATING MONTHLY PAYMENTS")
        amount = float(input("Enter total amount: "))
        number_of_years = float(input("Enter Number of years:"))
        payment = amount / (number_of_years * 12)
        print("Results:", payment)

    # else if uswer choose 7, program quit.
    elif userchoice == 7:
        print("Thank you. Goodbye")
        break
