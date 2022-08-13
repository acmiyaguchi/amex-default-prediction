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
  for_each = toset(["artifactregistry", "batch", "compute"])
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

// https://cloud.google.com/batch/docs/get-started
data "google_compute_default_service_account" "default" {
  depends_on = [google_project_service.services["compute"]]
}

// add roles to the service account
resource "google_project_iam_member" "default" {
  for_each = toset(["roles/batch.agentReporter", "roles/storage.admin"])
  project  = local.project_id
  role     = each.key
  member   = "serviceAccount:${data.google_compute_default_service_account.default.email}"
}

resource "google_compute_instance_template" "gpu" {
  for_each    = toset(["spot", "standard"])
  name_prefix = "amex-torch-gpu-${each.key}"

  machine_type = "n1-standard-4"

  // for k80
  region = "us-central1-a"

  tags = ["http-server", "https-server"]
  labels = {
    version = "1"
  }

  scheduling {
    preemptible        = each.key == "spot"
    provisioning_model = upper(each.key)
    automatic_restart  = false
  }

  // Create a new boot disk from an image
  disk {
    source_image = "projects/ml-images/global/images/c2-deeplearning-pytorch-1-12-cu113-v20220806-debian-10"
    auto_delete  = true
    boot         = true
    disk_type    = "pd-ssd"
    disk_size_gb = 100
  }

  network_interface {
    network = "default"
  }

  guest_accelerator {
    count = 1
    type  = "nvidia-tesla-k80"
  }

  lifecycle {
    create_before_destroy = true
  }
}

output "template_id" {
  value = {
    for key in ["spot", "standard"] : key => google_compute_instance_template.gpu[key].id
  }
}
