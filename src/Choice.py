# This is the file that allows tha user to use content based filtering or collaborative filtering.
import os
import sys

choice = input("Choose a way to execute: ")

choice_1 = "Content filtering"
choice_2 = "Collaborative filtering"
choice_3 = "Close"

choice_list = [choice_1, choice_2, choice_3]

counter = 0

while choice not in choice_list and counter < 5:
    choice = input("Choose a way to execute: ")
    counter = counter + 1
while choice not in choice_list and counter == 5:
    choice = input("RE CHOOSE A WAY LEME: ")
    counter = counter + 1
while choice not in choice_list and counter > 5:
    choice = input("RE FUGE RE ... RE BRO: ")
    counter = counter + 1

if choice == choice_2:
    os.system('python main.py')
if choice == choice_1:
    os.system('python second_try.py')
if choice == choice_3:
    sys.exit()

