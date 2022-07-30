terraform {
  backend "gcs" {
    bucket = "amex-default-prediction-2022-tfstate"
    prefix = "default"
  }
}

locals {
  project_id = "amex-default-prediction-2022"
  region     = "us-central1"
}

provider "google" {
  project = local.project_id
  region  = local.region
}

data "google_project" "project" {}

resource "google_container_registry" "registry" {
  location = "US"
}

resource "google_project_service" "services" {
  for_each = toset(["artifactregistry"])
  service  = "${each.key}.googleapis.com"
}

resource "google_storage_bucket" "data" {
  name     = "${local.project_id}-data"
  location = "US"
}

resource "google_artifact_registry_repository" "default" {
  location      = local.region
  repository_id = local.project_id
  format        = "DOCKER"
  depends_on    = [google_project_service.services["artifactregistry"]]
}
