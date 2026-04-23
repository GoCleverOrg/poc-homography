-- Database User Setup for poc-homography/hom
-- Run this as neondb_owner (superuser)
--
-- Users:
--   hom_migrate: DDL + DML on all tables (for migrations)
--   hom_app: DML on all tables, but camera_configs is SELECT only (runtime)
--   hom_infra: Only camera_configs table access (for infra project)

-- Create users (passwords are placeholders - set via script)
CREATE USER hom_migrate WITH PASSWORD 'MIGRATE_PASSWORD_PLACEHOLDER';
CREATE USER hom_app WITH PASSWORD 'APP_PASSWORD_PLACEHOLDER';
CREATE USER hom_infra WITH PASSWORD 'INFRA_PASSWORD_PLACEHOLDER';

-- Grant connect to database
GRANT CONNECT ON DATABASE hom TO hom_migrate, hom_app, hom_infra;

-- Grant schema usage
GRANT USAGE ON SCHEMA public TO hom_migrate, hom_app, hom_infra;

-- hom_migrate: Full DDL + DML privileges (for running migrations)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO hom_migrate;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO hom_migrate;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO hom_migrate;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO hom_migrate;

-- hom_app: DML on all tables (SELECT, INSERT, UPDATE, DELETE)
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO hom_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO hom_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO hom_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO hom_app;

-- hom_app: Revoke write access to camera_configs (read-only for this table)
REVOKE INSERT, UPDATE, DELETE ON camera_configs FROM hom_app;

-- hom_infra: Only camera_configs table access
GRANT SELECT, INSERT, UPDATE, DELETE ON camera_configs TO hom_infra;
-- No access to other tables (no grants needed - denied by default)
