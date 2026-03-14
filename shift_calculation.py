product = input('Введите наименование продукта: ')
plan = float(input('Введите план на смену, т: '))
fact = float(input('Введите фактический выпуск, т: '))

percent = (fact / plan * 100) if plan else 0
result = 'План выполнен' if fact >= plan else 'План не выполнен'
line = f'{product}; план={plan}; факт={fact}; выполнение={percent:.2f}%; {result}'

print(line)
with open('shift_results.txt', 'a', encoding='utf-8') as f:
    f.write(line + '\n')
