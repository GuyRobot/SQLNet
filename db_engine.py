import records
import re
from babel.numbers import parse_decimal, NumberFormatError

schema_re = re.compile(r'\((.+)\)')  # group (...) dfdf (...) group
num_re = re.compile(r'[-+]?\d*\.\d+|\d+')  # ? zero or one time appear of preceding character, * zero or several time
# appear of preceding character, catch something like -34.12, .23424

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']


class DBEngine:

    def __init__(self, fdb):
        self.db = records.Database(f'sqlite:///{fdb}').get_connection()

    def execute_query(self, table_id, query, *args, **kwargs):
        return self.execute(table_id, query.sel_index, query.agg_index, query.conditions, *args, **kwargs)

    def execute(self, table_id, select_index, aggregation_index, conditions, lower=True, return_query=False):
        if not table_id.startswith('table'):
            table_id = f'table_{table_id.replace("-", "_")}'
        table_info = self.db.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[
            0].sql.replace('\n', '')
        schema_str = schema_re.findall(table_info)[0]
        schema = {}
        for tup in schema_str.split(', '):
            c, t = tup.split()
            schema[c] = t
        select = f'col{select_index}'
        agg = agg_ops[aggregation_index]
        if agg:
            select = f'{agg}({select})'

        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            if lower and (isinstance(val, str)):
                val = val.lower()
            if schema[f'col{col_index}'] == 'real' and not isinstance(val, (int, float)):
                try:
                    # print('!!!!!!value of val is: ', val, 'type is: ', type(val))
                    # val = float(parse_decimal(val)) # somehow it generates error.
                    val = float(parse_decimal(val, locale='en_US'))
                    # print('!!!!!!After: val', val)
                except NumberFormatError as e:
                    try:
                        val = float(num_re.findall(val)[0])  # need to understand and debug this part
                    except:
                        # Although column is of number, selected one is not number. Do nothing in this case.
                        pass

            where_clause.append(f'col{col_index} {cond_ops[op]} :col{col_index}')
            where_map[f'col{col_index}'] = val

        where_str = ''
        if where_clause:
            where_str = f'WHERE {" AND ".join(where_clause)}'
        query = f'SELECT {select} AS result FROM {table_id} {where_str}'
        print(f"Query: {query}")
        out = self.db.query(query, **where_map)

        if return_query:
            return [o.result for o in out], query

        return [o.result for o in out]

    def execute_return_query(self, table_id, select_index, aggregation_index, conditions, lower=True):
        return self.execute(table_id, select_index, aggregation_index, conditions, lower=lower, return_query=True)

    def show_table(self, table_id):
        if not table_id.startswith('table'):
            table_id = f'table_{table_id.replace("-", "_")}'
        rows = self.db.query(f'SELECT * from {table_id}')
        print(rows.dataset)
