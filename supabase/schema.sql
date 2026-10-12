create schema if not exists raw;
create schema if not exists curated;
create schema if not exists ml;

create table if not exists raw.attrition_source (
  row_id bigint generated always as identity primary key,
  payload jsonb not null,
  loaded_at timestamptz not null default now()
);

create table if not exists curated.employee_attrition (
  employee_id bigint primary key,
  split text check (split in ('train', 'test')) not null,
  attrition boolean,
  features jsonb not null
);

create table if not exists ml.model_runs (
  run_id uuid primary key default gen_random_uuid(),
  model_name text not null,
  model_family text not null,
  dataset_version text not null,
  metrics jsonb not null,
  created_at timestamptz not null default now()
);

create table if not exists ml.predictions (
  prediction_id bigint generated always as identity primary key,
  run_id uuid references ml.model_runs(run_id),
  employee_id bigint not null,
  y_true boolean,
  y_score double precision,
  y_pred boolean,
  metadata jsonb not null default '{}'::jsonb
);
