# from pprint import pprint

# read file
weblog_file = open('./weblog.csv')
weblogs = weblog_file.readlines()
weblog_file.close()

number = 0
raw_number = 0
absolute_path = False
# write good data into a new file
web_logs = open('web_logs.csv', 'w')

for a_log in weblogs:
    raw_number += 1
    if a_log[0].isnumeric():
        fields = a_log.split(',')
        # pprint(fields)

        cmd_url = fields[2].split(' ')
        if cmd_url[1][0] != '/':
            print(fields[2])
            absolute_path = True
        else:
            number += 1
            web_logs.write(a_log)

web_logs.close()

# print statistic result
print('\n\nTotal number: ' + str(number))
print('Raw number: ' + str(raw_number))
print('Is absolute path? ' + ('yes' if absolute_path else 'no'))
