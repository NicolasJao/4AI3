from datetime import date, timedelta

# 1. Swap first and last elements of a list
fruits = ["orange", "apple", "banana", "pear", "watermelon"]
output = [fruits[-1]]               # last element first
for i in range(1, len(fruits) - 1): # middle elements only
    output.append(fruits[i])
output.append(fruits[0])            # first element goes last
print(output)

# 2. Print today's, yesterday's, and tomorrow's date
today = date.today()
yesterday = today - timedelta(days=1)
tomorrow = today + timedelta(days=1)
date_format = "%d %b %Y"
print(f"Today's date: {today.strftime(date_format)}")
print(f"Yesterday's date: {yesterday.strftime(date_format)}")
print(f"Tomorrow's date: {tomorrow.strftime(date_format)}")

# 3. Sort a list of numbers in descending order
def sort_descending(numbers):
    n = len(numbers)
    for i in range(n):
        for j in range(n - i - 1):
            if numbers[j] < numbers[j + 1]:
                numbers[j], numbers[j + 1] = numbers[j + 1], numbers[j]
    return numbers
example = [31, 100, 2, 64, 87]
example2 = [1, 2, 3, 4, 5, 6, 7]
print(sort_descending(example))
print(sort_descending(example2))

# 4. Find the most occurring word and its count from a string
text = "I am enrolled in a McMaster degree program. There are a number of McMaster degree programs listed in top 100 degree programs of the world."
cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
words = cleaned.split()

word_counts = {}
for word in words:
    word_counts[word] = word_counts.get(word, 0) + 1

most_common_word = None
highest_count = 0
for word, count in word_counts.items():
    if count > highest_count:
        highest_count = count
        most_common_word = word

print("Most occurring word: " + most_common_word)
print("Word Count: " + str(highest_count))
