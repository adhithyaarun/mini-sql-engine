import csv
import sys
import sqlparse as sql


class SQL_Engine():
    def __init__(self, path):
        self.path = path
        self.AGGREGATE = ['SUM', 'AVG', 'MAX', 'MIN']
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

    def process_join(self):
        try:
            if len(self.query['tables']) > 1 and len(self.query['conditions']) > 0:
                print(self.query['conditions'])
        except:
            pass

    def table_check(self):
        for table in self.query['tables']:
            if str(table) not in list(self.tables.keys()):
                print('Table {} does not exist.'.format(str(table)))
                return False
        return True

    def standardize_column(self):
        try:
            flag = False
            for table in self.query['tables']:
                if len(self.query['distinct']) > 0:
                    for index, column in enumerate(self.query['distinct']):
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
                                    print('Table {0} does not exist.'.format(
                                        str(specified[0])))
                                    return False
                        else:
                            if len(list(self.query['tables'])) > 1 and list(filter(lambda a: str(column) in self.tables[a]['column_names'], self.tables.keys())) == list(self.query['tables']):
                                print('Ambiguous Query.')
                                return False
                            else:
                                if str(column) in self.tables[str(table)]['column_names']:
                                    self.query['distinct'][index] = str(
                                        table) + '.' + str(column)
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
                                pass
                            else:
                                if specified[0] in self.query['tables']:
                                    print('Column {0} not in {1}.'.format(
                                        str(column), str(specified[0])))
                                    return False
                                else:
                                    print('Table {0} does not exist.'.format(
                                        str(specified[0])))
                                    return False
                        else:
                            if len(list(self.query['tables'])) > 1 and list(filter(lambda a: str(column) in self.tables[a]['column_names'], self.tables.keys())) == list(self.query['tables']):
                                print('Ambiguous Query.')
                                return False
                            else:
                                if str(column) in self.tables[str(table)]['column_names']:
                                    self.query['aggregations'][list(self.query['aggregations'].keys())[
                                        0]][index] = str(table) + '.' + str(column)
                                else:
                                    print('Column {0} not in {1}'.format(
                                        str(column), str(table)))
                                    return False
                elif len(self.query['columns']) > 0:
                    if self.query['columns'] == ['all']:
                        self.query['columns'] = list(
                            self.tables[str(table)]['columns'].keys())
                    for index, column in enumerate(self.query['columns']):
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
                                    print('Table {0} does not exist.'.format(
                                        str(specified[0])))
                                    return False
                        else:
                            if len(list(self.query['tables'])) > 1 and list(filter(lambda a: str(column) in self.tables[a]['column_names'], self.tables.keys())) == list(self.query['tables']):
                                print('Ambiguous Query.')
                                return False
                            else:
                                if str(column) in self.tables[str(table)]['column_names']:
                                    self.query['columns'][index] = str(
                                        table) + '.' + str(column)
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
                                                    print('Table {0} does not exist.'.format(
                                                        str(specified[0])))
                                                    return False
                                        else:
                                            if len(list(self.query['tables'])) > 1 and list(filter(lambda a: str(column) in self.tables[a]['column_names'], self.tables.keys())) == list(self.query['tables']):
                                                print('Ambiguous Query.')
                                                return False
                                            else:
                                                if str(column) in self.tables[str(table)]['column_names']:
                                                    operation[index] = str(
                                                        table) + '.' + str(column)
                                                else:
                                                    print('Column {0} not in {1}'.format(
                                                        str(column), str(table)))
                                                    return False
                                    operation.append(operator)
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

    def process_query(self):
        try:
            if self.table_check() and self.standardize_column():
                if len(self.query['distinct']) > 0:

                elif len(self.query['aggregations']) > 0:
                    aggregation = list(
                        self.query['aggregations'].keys())[0]
                    for column in self.query['aggregations'][list(self.query['aggregations'].keys())[0]]:
                        if '.' in str(column):
                            specified = str(column).split('.')
                            if specified[0] in self.query['tables'] and specified[1] in self.tables[specified[0]]['column_names']:
                                print(self.tables[specified[0]]
                                      ['columns'][specified[1]])
                                self.query['aggregations'][list(self.query['aggregations'].keys())[
                                    0]].remove(column)
                            else:
                                print('Invalid Query.')
                                return
                        else:
                            if len(list(self.query['tables'])) > 1 and list(filter(lambda a: str(column) in self.tables[a]['column_names'], self.tables.keys())) == list(self.query['tables']):
                                print('Ambiguous Query.')
                                return
                            else:
                                if str(column) in self.tables[str(table)]['column_names']:
                                    print(self.tables[str(table)]
                                          ['columns'][str(column)])
                                else:
                                    print('Column {0} not in {1}'.format(
                                        str(column), str(table)))
                elif len(self.query['columns']) > 0:
                    if self.query['columns'] == ['all']:
                        self.query['columns'] = list(
                            self.tables[str(table)]['columns'].keys())
                    for column in self.query['columns']:
                        if '.' in str(column):
                            specified = str(column).split('.')
                            if specified[0] in self.query['tables'] and specified[1] in self.tables[specified[0]]['column_names']:
                                print(self.tables[specified[0]]
                                      ['columns'][specified[1]])
                                self.query['columns'].remove(column)
                            else:
                                print('Invalid Query.')
                                return
                        else:
                            if len(list(self.query['tables'])) > 1 and list(filter(lambda a: str(column) in self.tables[a]['column_names'], self.tables.keys())) == list(self.query['tables']):
                                print('Ambiguous Query.')
                                return
                            else:
                                if str(column) in self.tables[str(table)]['column_names']:
                                    print(self.tables[str(table)]
                                          ['columns'][str(column)])
                                else:
                                    print('Column {0} not in {1}'.format(
                                        str(column), str(table)))
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
            'conditions': []
        }

        if query_type == 'exit' or query_type == 'quit':
            self.quit = True
            return (True, None, parsed)
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
                    if extracted.split('(')[0].upper() in self.AGGREGATE:
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
                return (False, 'Invalid Query.', [], [], [])

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
                    filter(lambda a: str(a) != ' ', query.tokens))

                (success, message, self.query) = self.parse_query(query_list)

                if self.quit is True:
                    break

                self.process_query()

                if success is True:
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

# def process_query(self):
    #     try:
    #         if self.table_check() and self.standardize_column():
    #             print(self.query)
    #             for table in self.query['tables']:
    #                 if len(self.query['distinct']) > 0:
    #                     for column in self.query['distinct']:
    #                         if '.' in str(column):
    #                             specified = str(column).split('.')
    #                             if specified[0] in self.query['tables'] and specified[1] in self.tables[specified[0]]['column_names']:
    #                                 print(self.tables[specified[0]]
    #                                       ['columns'][specified[1]])
    #                                 self.query['distinct'].remove(column)
    #                             else:
    #                                 print('Invalid Query.')
    #                                 return
    #                         else:
    #                             if len(list(self.query['tables'])) > 1 and list(filter(lambda a: str(column) in self.tables[a]['column_names'], self.tables.keys())) == list(self.query['tables']):
    #                                 print('Ambiguous Query.')
    #                                 return
    #                             else:
    #                                 if str(column) in self.tables[str(table)]['column_names']:
    #                                     print(self.tables[str(table)]
    #                                           ['columns'][str(column)])
    #                                 else:
    #                                     print('Column {0} not in {1}'.format(
    #                                         str(column), str(table)))
    #                 elif len(self.query['aggregations']) > 0:
    #                     aggregation = list(
    #                         self.query['aggregations'].keys())[0]
    #                     for column in self.query['aggregations'][list(self.query['aggregations'].keys())[0]]:
    #                         if '.' in str(column):
    #                             specified = str(column).split('.')
    #                             if specified[0] in self.query['tables'] and specified[1] in self.tables[specified[0]]['column_names']:
    #                                 print(self.tables[specified[0]]
    #                                       ['columns'][specified[1]])
    #                                 self.query['aggregations'][list(self.query['aggregations'].keys())[
    #                                     0]].remove(column)
    #                             else:
    #                                 print('Invalid Query.')
    #                                 return
    #                         else:
    #                             if len(list(self.query['tables'])) > 1 and list(filter(lambda a: str(column) in self.tables[a]['column_names'], self.tables.keys())) == list(self.query['tables']):
    #                                 print('Ambiguous Query.')
    #                                 return
    #                             else:
    #                                 if str(column) in self.tables[str(table)]['column_names']:
    #                                     print(self.tables[str(table)]
    #                                           ['columns'][str(column)])
    #                                 else:
    #                                     print('Column {0} not in {1}'.format(
    #                                         str(column), str(table)))
    #                 elif len(self.query['columns']) > 0:
    #                     if self.query['columns'] == ['all']:
    #                         self.query['columns'] = list(
    #                             self.tables[str(table)]['columns'].keys())
    #                     for column in self.query['columns']:
    #                         if '.' in str(column):
    #                             specified = str(column).split('.')
    #                             if specified[0] in self.query['tables'] and specified[1] in self.tables[specified[0]]['column_names']:
    #                                 print(self.tables[specified[0]]
    #                                       ['columns'][specified[1]])
    #                                 self.query['columns'].remove(column)
    #                             else:
    #                                 print('Invalid Query.')
    #                                 return
    #                         else:
    #                             if len(list(self.query['tables'])) > 1 and list(filter(lambda a: str(column) in self.tables[a]['column_names'], self.tables.keys())) == list(self.query['tables']):
    #                                 print('Ambiguous Query.')
    #                                 return
    #                             else:
    #                                 if str(column) in self.tables[str(table)]['column_names']:
    #                                     print(self.tables[str(table)]
    #                                           ['columns'][str(column)])
    #                                 else:
    #                                     print('Column {0} not in {1}'.format(
    #                                         str(column), str(table)))
    #     except:
    #         print('Invalid Query.')
    #         return