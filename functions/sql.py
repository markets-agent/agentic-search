from ypostgres_lib import run_static_dql


def get_postgres_db_schema():
    res = run_static_dql(
        """
-- get all enum types and their possible values from the system catalog 
WITH enum_values AS (
    SELECT 
        n.nspname AS schema_name,
        t.typname AS enum_type,   -- name of the enum type
        e.enumlabel AS enum_value -- individual enum values
    FROM pg_type t
    JOIN pg_enum e ON t.oid = e.enumtypid -- join to get enum values
    JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace -- get schema info
    WHERE n.nspname = 'public'
),
-- collect all CHECK constraints and the columns they apply to
check_constraints AS (
    SELECT 
        tc.table_name,
        cc.column_name,
        chk.check_clause
    FROM information_schema.check_constraints chk
    JOIN information_schema.constraint_column_usage cc 
        ON chk.constraint_name = cc.constraint_name
    JOIN information_schema.table_constraints tc 
        ON chk.constraint_name = tc.constraint_name
    WHERE tc.table_schema = 'public'
),
-- get all indexes defined on tables
table_indexes AS (
    SELECT 
        schemaname,
        tablename,
        indexname,
        indexdef -- complete index definition (includes columns and conditions)
    FROM pg_indexes
    WHERE schemaname = 'public'
)
-- main query combining all schema information
SELECT 
    -- basic column information
    c.table_name,
    c.column_name,
    c.column_default,
    c.data_type,
    c.is_nullable,
    -- detailed type information
    c.character_maximum_length, -- for varchar/char types
    c.numeric_precision,        -- for numeric types
    c.numeric_scale,            -- for numeric types
    -- handle custom enum types by collecting their values
    CASE 
        WHEN c.data_type = 'USER-DEFINED' THEN 
            (SELECT string_agg(ev.enum_value, ', ')
                FROM enum_values ev 
                WHERE ev.enum_type = c.udt_name)
        ELSE NULL 
    END as enum_values,
    -- include any CHECK constraints
    chk.check_clause as check_constraints,
    -- aggregate all indexes for the table
    (
        SELECT string_agg(indexdef, ' | ')
        FROM table_indexes i 
        WHERE i.tablename = c.table_name
    ) as table_indexes,
    -- get primary key information
    (
        SELECT string_agg(kcu.column_name, ', ')
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu 
            ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_name = c.table_name 
        AND tc.constraint_type = 'PRIMARY KEY'
    ) as primary_key,
    -- get foreign key relationships
    (
        SELECT string_agg(
            format(
                '%s REFERENCES %s(%s)',
                kcu.column_name, -- source column
                ccu.table_name,  -- referenced table
                ccu.column_name  -- referenced column
            ),
            ' | '
        )
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu 
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage ccu 
            ON tc.constraint_name = ccu.constraint_name
        WHERE tc.table_name = c.table_name 
        AND tc.constraint_type = 'FOREIGN KEY'
    ) as foreign_keys
FROM information_schema.columns c
-- join with check constraints (if any exist for the column)
LEFT JOIN check_constraints chk 
    ON c.table_name = chk.table_name 
    AND c.column_name = chk.column_name
WHERE c.table_schema = 'public'
-- order results by table name and column position
ORDER BY c.table_name, c.ordinal_position;"""
    )
    return res


def is_valid_query(input_obj):
    return input_obj["is_valid"]
