# importing the libraries
import re

string = 'tiger is the national animal of India.'
pattern = 'tiger'
# match in the beginning of the string
result = re.match(pattern, string)
print (result)
print (result.group(0)) # prints only the match
pattern2 = 'national'
result2 = re.match(pattern2, string)
print (result2)
# match anywhere in the string
pattern3 = 'national'
result3 = re.search(pattern3, string)
print (result3)
print (result3.group(0)) # prints only the search
# find anywhere and store as a list
string2 = 'tiger is the national animal of India and national sports is hockey.'
result4 = re.findall(pattern3, string2)
print (result4)
# print (result4.group(0)) .... 'list' object has no attribute 'group'
# find anywhere and get the index
result5 = re.finditer(pattern3, string2)
# print (result5) # callable_iterator object at given memory location
for n in result5: # immidiate iteration is required
    print(n.start(), n.end())
    
# wildcard expressions
string3 = 'I was born on 06-12-1992, fully vaccinated on 09-07-1997, still i was infected on 30-02-2007'
pattern4 = '\d{2}-\d{2}-\d{4}'
result6 = re.findall(pattern4, string3)
print(result6)
result6 = re.finditer(pattern4, string3)
# print (result6) # callable_iterator object at given memory location
for n in result6: # immidiate iteration is required
    print(n.start(), n.end())
    
# substitution
print(re.sub(pattern4, 'NaN', string3))

# Taking care of the spaces
string4 = 'The team started on 26 - 10 - 2018 and reached the destination on 27 - 10 - 2018'
pattern5 = '\d{2}\s-\s\d{2}\s-\s\d{4}'
result7 = re.findall(pattern5, string4)
print(result7)
result7 = re.finditer(pattern5, string4)
# print (result6) # callable_iterator object at given memory location
for n in result7: # immidiate iteration is required
    print(n.start(), n.end())