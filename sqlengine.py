import csv
import sys
import sqlparse as sql
import itertools
import statistics
import os


class SQL_Engine():
    def __init__(self, path):
        self.path = path
        self.AGGREGATE = {
            'SUM': sum,
            'AVG': statistics.mean,
            'MAX': max,
            'MIN': min
        }
        self.RELATIONAL_OPERATORS = ['<', '>', '<=', '>=', '=', '<>']
        self.history = []
        self.quit = False
        self.query = None

        (self.table_index, self.tables) = self.parse_meta_data(
            '{0}/metadata.txt'.format(self.path))

        for table in self.table_index:
            self.tables[table] = self.read_data(
                '{0}/{1}.csv'.format(self.path, table), self.tables[table])

    @staticmethod
    def parse_meta_data(filepath=None):
        if filepath is None:
            print('ERROR: Please provide path to metadata file.')
            return None
        else:
            with open(filepath, 'r') as file:
                lines = file.read().splitlines()
                space = ' '
                n = '\n'

                intermediate = space.join(lines).split('<end_table>')
                intermediate = n.join(intermediate).split('<begin_table>')
                intermediate = space.join(intermediate).split(' ')
                intermediate = list(filter(lambda a: a != '', intermediate))

                start = True
                table_index = {}
                tables = {}
                table = {}
                table_counter = 0
                attribute_counter = 0

                for word in intermediate:
                    if start is True:
                        table_index[word] = table_counter
                        attribute_counter = 0
                        table = {
                            'table_name': word,
                            'column_names': [],
                            'column_index': {},
                            'columns': {},
                            'records': []
                        }
                        start = False
                        table_counter = table_counter + 1
                    elif word == '\n':
                        tables[table['table_name']] = table
                        start = True
                    else:
                        table['column_index'][word] = attribute_counter
                        table['column_names'].append(word)
                        attribute_counter = attribute_counter + 1
                        table['columns'][word] = []
                return (table_index, tables)

    @staticmethod
    def read_data(filepath=None, table=None):
        if filepath is None:
            print('ERROR: Please provide path to data file.')
            return table
        elif table is None:
            print('ERROR: Please provide table data structure.')
            return table
        else:
            with open(filepath, 'r') as file:
                reader = csv.reader(file)

                for row in reader:
                    record = []
                    for i in range(len(row)):
                        table['columns'][table['column_names']
                                         [i]].append(int(row[i]))
                        record.append((table['column_names'][i], int(row[i])))
                    table['records'].append(dict(record))
            return table

    def extract_standardised(self, raw):
        queries = sql.split(raw)
        parsed = []
        for query in queries:
            formatted_query = sql.format(
                query, keyword_case='upper')
            tokens = sql.parse(formatted_query)[0]
            parsed.append(tokens)
        return parsed

    def table_check(self):
        for table in self.query['tables']:
            if str(table) not in list(self.tables.keys()):
                print('Table {} does not exist.'.format(str(table)))
                return False
        return True

    def standardize_column(self):
        try:
            flag = False
            if len(self.query['distinct']) > 0:
                if self.query['distinct'] == ['all']:
                    cols = []
                    for table in self.query['tables']:
                        cols.extend(list(
                            map(lambda a: table + '.' + a, list(self.tables[str(table)]['columns'].keys()))))
                    self.query['distinct'] = cols
                else:
                    for index, column in enumerate(self.query['distinct']):
                        if '.' in str(column):
                            specified = str(column).split('.')
                            if specified[0] in self.query['tables'] and specified[1] in self.tables[specified[0]]['column_names']:
                                self.query['distinct'][index] = str(column)
                            else:
                                if specified[0] in self.query['tables']:
                                    print('Column {0} not in {1}.'.format(
                                        str(column), str(specified[0])))
                                    return False
                                else:
                                    print('Table {0} not specified.'.format(
                                        str(specified[0])))
                                    return False
                        else:
                            if len(list(self.query['tables'])) > 1 and list(filter(lambda a: str(column) in self.tables[a]['column_names'], self.tables.keys())) == list(self.query['tables']):
                                print('Ambiguous Query.')
                                return False
                            else:
                                for table in self.query['tables']:
                                    if str(column) in self.tables[str(table)]['column_names']:
                                        self.query['distinct'][index] = str(
                                            table) + '.' + str(column)
                                        break
                                    else:
                                        print('Column {0} not in {1}'.format(
                                            str(column), str(table)))
                                        return False
            elif len(self.query['aggregations']) > 0:
                aggregation = list(self.query['aggregations'].keys())[0]
                for index, column in enumerate(self.query['aggregations'][list(self.query['aggregations'].keys())[0]]):
                    if '.' in str(column):
                        specified = str(column).split('.')
                        if specified[0] in self.query['tables'] and specified[1] in self.tables[specified[0]]['column_names']:
                            self.query['aggregations'][list(self.query['aggregations'].keys())[
                                0]][index] = str(column)
                        else:
                            if specified[0] in self.query['tables']:
                                print('Column {0} not in {1}.'.format(
                                    str(column), str(specified[0])))
                                return False
                            else:
                                print('Table {0} not specified.'.format(
                                    str(specified[0])))
                                return False
                    else:
                        if len(list(self.query['tables'])) > 1 and list(filter(lambda a: str(column) in self.tables[a]['column_names'], self.tables.keys())) == list(self.query['tables']):
                            print('Ambiguous Query.')
                            return False
                        else:
                            for table in self.query['tables']:
                                if str(column) in self.tables[str(table)]['column_names']:
                                    self.query['aggregations'][list(self.query['aggregations'].keys())[
                                        0]][index] = str(table) + '.' + str(column)
                                    break
                                else:
                                    print('Column {0} not in {1}'.format(
                                        str(column), str(table)))
                                    return False
            elif len(self.query['columns']) > 0:
                if self.query['columns'] == ['all']:
                    cols = []
                    for table in self.query['tables']:
                        cols.extend(list(
                            map(lambda a: table + '.' + a, list(self.tables[str(table)]['columns'].keys()))))
                    self.query['columns'] = cols
                else:
                    for index, column in enumerate(self.query['columns']):
                        if '.' in str(column):
                            specified = str(column).split('.')
                            if specified[0] in self.query['tables'] and specified[1] in self.tables[specified[0]]['column_names']:
                                self.query['columns'][index] = str(column)
                            else:
                                if specified[0] in self.query['tables']:
                                    print('Column {0} not in {1}.'.format(
                                        str(column), str(specified[0])))
                                    return False
                                else:
                                    print('Table {0} not specified.'.format(
                                        str(specified[0])))
                                    return False
                        else:
                            if len(list(self.query['tables'])) > 1 and list(filter(lambda a: str(column) in self.tables[a]['column_names'], self.tables.keys())) == list(self.query['tables']):
                                print('Ambiguous Query.')
                                return False
                            else:
                                for table in self.query['tables']:
                                    if str(column) in self.tables[str(table)]['column_names']:
                                        self.query['columns'][index] = str(
                                            table) + '.' + str(column)
                                        break
                                    else:
                                        print('Column {0} not in {1}'.format(
                                            str(column), str(table)))
                                        return False
            if len(self.query['conditions']) > 0:
                operators = ['<>', '<=', '>=', '<', '>', '=']
                for ind, condition in enumerate(self.query['conditions']):
                    flag = False
                    if str(condition) in ['AND', 'OR']:
                        flag = True
                        continue
                    else:
                        for operator in operators:
                            if len(condition.split(operator)) > 1:
                                operation = condition.split(operator)
                                for index, column in enumerate(operation):
                                    if str(column).isdigit() is True:
                                        break
                                    if '.' in str(column):
                                        specified = str(column).split('.')
                                        if specified[0] in self.query['tables'] and specified[1] in self.tables[specified[0]]['column_names']:
                                            pass
                                        else:
                                            if specified[0] in self.query['tables']:
                                                print('Column {0} not in {1}.'.format(
                                                    str(column), str(specified[0])))
                                                return False
                                            else:
                                                print('Table {0} not specified.'.format(
                                                    str(specified[0])))
                                                return False
                                    else:
                                        if len(list(self.query['tables'])) > 1 and list(filter(lambda a: str(column) in self.tables[a]['column_names'], self.tables.keys())) == list(self.query['tables']):
                                            print('Ambiguous Query.')
                                            return False
                                        else:
                                            for table in self.query['tables']:
                                                if str(column) in self.tables[str(table)]['column_names']:
                                                    operation[index] = str(
                                                        table) + '.' + str(column)
                                                else:
                                                    print('Column {0} not in {1}'.format(
                                                        str(column), str(table)))
                                                    return False
                                operation.append(operator)
                                operand1 = operation[0].split('.')
                                operand2 = operation[1].split('.')
                                if len(self.query['tables']) > 1 and len(operand1) > 0 and len(operand2) > 0:
                                    if operand1[0] != operand2[0]:
                                        self.query['join'] = True
                                        # If both are asked, exclude 2nd, if one is asked exclude other
                                        self.query['exclude'].append(
                                            '.'.join(operand2))

                                self.query['conditions'][ind] = operation
                                flag = True
                                break
                        if flag is False:
                            print('Invalid operation')
                            return False

            return True
        except:
            print('Invalid Query.')
            return False

    def aggregation_handler(self, function, column):
        return self.AGGREGATE[function](column)

    def extract_records(self):
        table_recs = []
        order = []
        records = []
        mapping = {}
        i = 0
        if len(self.query['tables']) > 1:
            for table in self.query['tables']:
                table_recs.append(
                    list(map(lambda a: tuple(a.values()), self.tables[table]['records'])))
                order = list(map(lambda a: table + '.' + a,
                                 self.tables[table]['column_names']))
                index = list(range(len(order)))
                index = list(map(lambda a: tuple((i, a)), index))
                i += 1
                mapping.update(dict(zip(order, index)))
            records = list(itertools.product(*table_recs))
        else:
            records = list(map(lambda a: tuple(a.values()),
                               self.tables[self.query['tables'][0]]['records']))
            order = list(map(lambda a: self.query['tables'][0] + '.' + a,
                             self.tables[self.query['tables'][0]]['column_names']))
            index = list(range(len(order)))
            index = list(map(lambda a: tuple((i, a)), index))
            mapping.update(dict(zip(order, index)))

        approved = records
        if len(self.query['conditions']) > 0:
            approved = []
            operation = None
            passed_now = []
            for condition in self.query['conditions']:
                if condition not in ['OR', 'AND']:
                    if '.' in condition[0] and '.' in condition[1]:
                        col1 = condition[0]
                        col2 = condition[1]
                        op = condition[2]
                        for row in records:
                            index_1 = list(mapping[col1])
                            index_2 = list(mapping[col2])
                            if op == '<>':
                                if len(self.query['tables']) > 1:
                                    if row[index_1[0]][index_1[1]] != row[index_2[0]][index_2[1]]:
                                        passed_now.append(row)
                                else:
                                    if row[index_1[1]] != row[index_2[1]]:
                                        passed_now.append(row)
                            elif op == '<=':
                                if len(self.query['tables']) > 1:
                                    if row[index_1[0]][index_1[1]] <= row[index_2[0]][index_2[1]]:
                                        passed_now.append(row)
                                else:
                                    if row[index_1[1]] <= row[index_2[1]]:
                                        passed_now.append(row)
                            elif op == '>=':
                                if len(self.query['tables']) > 1:
                                    if row[index_1[0]][index_1[1]] >= row[index_2[0]][index_2[1]]:
                                        passed_now.append(row)
                                else:
                                    if row[index_1[1]] >= row[index_2[1]]:
                                        passed_now.append(row)
                            elif op == '<':
                                if len(self.query['tables']) > 1:
                                    if row[index_1[0]][index_1[1]] < row[index_2[0]][index_2[1]]:
                                        passed_now.append(row)
                                else:
                                    if row[index_1[1]] < row[index_2[1]]:
                                        passed_now.append(row)
                            elif op == '>':
                                if len(self.query['tables']) > 1:
                                    if row[index_1[0]][index_1[1]] > row[index_2[0]][index_2[1]]:
                                        passed_now.append(row)
                                else:
                                    if row[index_1[1]] > row[index_2[1]]:
                                        passed_now.append(row)
                            elif op == '=':
                                if len(self.query['tables']) > 1:
                                    if row[index_1[0]][index_1[1]] == row[index_2[0]][index_2[1]]:
                                        passed_now.append(row)
                                else:
                                    if row[index_1[1]] == row[index_2[1]]:
                                        passed_now.append(row)
                    else:
                        col1 = condition[0]
                        col2 = condition[1]
                        op = condition[2]
                        for row in records:
                            index_1 = list(mapping[col1])
                            if op == '<>':
                                if len(self.query['tables']) > 1:
                                    if row[index_1[0]][index_1[1]] != int(col2):
                                        passed_now.append(row)
                                else:
                                    if row[index_1[1]] != int(col2):
                                        passed_now.append(row)

                            elif op == '<=':
                                if len(self.query['tables']) > 1:
                                    if row[index_1[0]][index_1[1]] <= int(col2):
                                        passed_now.append(row)
                                else:
                                    if row[index_1[1]] <= int(col2):
                                        passed_now.append(row)
                            elif op == '>=':
                                if len(self.query['tables']) > 1:
                                    if row[index_1[0]][index_1[1]] >= int(col2):
                                        passed_now.append(row)
                                else:
                                    if row[index_1[1]] >= int(col2):
                                        passed_now.append(row)
                            elif op == '<':
                                if len(self.query['tables']) > 1:
                                    if row[index_1[0]][index_1[1]] < int(col2):
                                        passed_now.append(row)
                                else:
                                    if row[index_1[1]] < int(col2):
                                        passed_now.append(row)
                            elif op == '>':
                                if len(self.query['tables']) > 1:
                                    if row[index_1[0]][index_1[1]] > int(col2):
                                        passed_now.append(row)
                                else:
                                    if row[index_1[1]] > int(col2):
                                        passed_now.append(row)
                            elif op == '=':
                                if len(self.query['tables']) > 1:
                                    if row[index_1[0]][index_1[1]] == int(col2):
                                        passed_now.append(row)
                                else:
                                    if row[index_1[1]] == int(col2):
                                        passed_now.append(row)
                else:
                    approved = passed_now
                    passed_now = []
                    operation = condition
            if operation is not None:
                if operation == 'OR':
                    approved = list(set(approved).union(set(passed_now)))
                elif operation == 'AND':
                    approved = list(
                        set(approved).intersection(set(passed_now)))
            else:
                approved = passed_now
        return (mapping, approved)

    def projected_columns(self, column_names, mapping, approved):
        names = tuple(column_names)
        projected = []
        for row in approved:
            record = []
            for name in column_names:
                if len(self.query['tables']) > 1:
                    record.append(row[mapping[name][0]][mapping[name][1]])
                else:
                    record.append(row[mapping[name][1]])
            projected.append(tuple(record))
        return (names, projected)

    def project(self, records, names, aggregation=False):
        if records is None:
            print('Empty set.')
        else:
            skip = []
            for i in range(len(names)):
                if self.query['join'] is True and names[i] in self.query['exclude']:
                    skip.append(i)
                    continue
                if i == len(names) - 1:
                    print('{0}'.format(names[i]), end='')
                else:
                    print('{0}, '.format(names[i]), end='')

            print('')
            if aggregation is True:
                print(records)
            else:
                for i in range(len(records)):
                    for j in range(len(records[i])):
                        if self.query['join'] is True and j in skip:
                            continue
                        if j == len(records[i]) - 1:
                            print('{0}'.format(records[i][j]), end='')
                        else:
                            print('{0}, '.format(records[i][j]), end='')
                    print('')

    def process_query(self):
        try:
            result = None
            if self.table_check() and self.standardize_column():
                (mapping, approved) = self.extract_records()

                if len(self.query['distinct']) > 0:
                    (names, projected) = self.projected_columns(
                        self.query['distinct'], mapping, approved)
                    if len(projected) > 0:
                        result = list(set(projected))
                    self.project(result, names)
                elif len(self.query['aggregations']) > 0:
                    column = self.query['aggregations'][list(
                        self.query['aggregations'].keys())[0]][0]
                    (names, projected) = self.projected_columns(
                        [column], mapping, approved)
                    columns = list(map(lambda a: a[0], projected))
                    if len(columns) > 0:
                        result = self.aggregation_handler(list(self.query['aggregations'].keys())[
                            0], columns)
                    self.project(result, ['{0}({1})'.format(
                        list(self.query['aggregations'].keys())[0], column)], True)
                elif len(self.query['columns']) > 0:
                    (names, projected) = self.projected_columns(
                        self.query['columns'], mapping, approved)
                    if len(projected) > 0:
                        result = list(projected)
                    self.project(result, names)
            return result
        except:
            print('Invalid Query.')
            return

    def parse_query(self, query):
        query_type = str(query[0])
        parsed = {
            'join': False,
            'distinct': [],
            'aggregations': {},
            'columns': [],
            'tables': [],
            'conditions': [],
            'exclude': []
        }

        if query_type == 'exit' or query_type == 'quit':
            self.quit = True
            return (True, None, parsed)
        elif query_type == 'history':
            for query in self.history:
                print(query)
            return (True, 'HISTORY', parsed)
        elif query_type == 'clear':
            os.system('clear')
            return (True, 'CLEAR', parsed)
        else:
            if query_type == 'SELECT':
                ind = 1

                if str(query[ind]) == 'DISTINCT':
                    ind += 1
                    if len(str(query[ind]).split(',')) > 1:
                        parsed['distinct'] = list(query[ind].get_identifiers())
                    else:
                        parsed['distinct'] = [str(query[ind])]
                elif str(query[ind]) == '*':
                    parsed['columns'] = ['all']
                else:
                    if len(str(query[ind]).split(',')) > 1:
                        parsed['columns'] = list(query[ind].get_identifiers())
                    else:
                        parsed['columns'] = [str(query[ind])]

                if len(parsed['columns']) == 1:
                    extracted = parsed['columns'][0].split(')')[0]
                    if extracted.split('(')[0].upper() in list(self.AGGREGATE.keys()):
                        parsed['aggregations'][extracted.split('(')[0].upper()] = [
                            extracted.split('(')[1]]
                        parsed['columns'] = []
                    elif extracted.split('(')[0].upper() == 'DISTINCT':
                        parsed['distinct'] = [extracted.split('(')[1]]
                        parsed['columns'] = []

                ind += 1

                if str(query[ind]) != 'FROM':
                    message = 'Missing FROM keyword.'
                    return (False, message, parsed)

                ind += 1

                if len(str(query[ind]).split(',')) > 1:
                    parsed['tables'] = list(
                        map(lambda a: str(a), list(query[ind].get_identifiers())))
                else:
                    parsed['tables'] = [str(query[ind])]

                ind += 1
                cond = []

                if len(parsed['distinct']) == 0 and len(query) > 4:
                    cond = str(query[ind]).split(' ')[1:]
                    start = 0
                    end = 0
                    for i in range(len(cond)):
                        if cond[i] == 'OR' or cond[i] == 'AND':
                            end = i
                            parsed['conditions'].append(
                                ''.join(cond[start:end]))
                            parsed['conditions'].append(cond[i])
                            start = end + 1
                    parsed['conditions'].append(
                        ''.join(cond[start:]).split(';')[0])
                return (True, None, parsed)
            else:
                return (False, 'Invalid Query.', {})

    def run(self):
        while(True):
            print('\nsql>> ', end='')
            success = False
            message = ''
            parsed = None
            raw_query = input()
            queries = self.extract_standardised(raw_query)

            for query in queries:
                query_list = list(
                    filter(lambda a: str(a) != ' ' and str(a) != ';', query.tokens))

                (success, message, self.query) = self.parse_query(query_list)

                if self.quit is True:
                    break
                elif message == 'HISTORY' or message == 'CLEAR':
                    continue

                if success is True:
                    self.process_query()
                    self.history.append(str(query))
                else:
                    print(message)

            if self.quit is True:
                break


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('ERROR: Please provide path to data directory.')
        exit(1)
    else:
        engine = SQL_Engine(sys.argv[1])
        engine.run()
