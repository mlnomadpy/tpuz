"""
Google Cloud Secret Manager integration.

Secrets are stored in GCP and accessed by the TPU VM directly via its
service account — no secrets transit through the local machine.

Setup (one-time):
    # Enable the API
    gcloud services enable secretmanager.googleapis.com

    # Store a secret
    echo -n "your-api-key" | gcloud secrets create WANDB_API_KEY --data-file=-

    # Grant TPU VM access (uses default compute service account)
    gcloud secrets add-iam-policy-binding WANDB_API_KEY \
        --member="serviceAccount:$(gcloud iam service-accounts list --format='value(email)' --filter='Compute Engine')" \
        --role="roles/secretmanager.secretAccessor"

Usage:
    from tpuz.secrets import SecretManager

    sm = SecretManager(project="my-project")
    sm.create("WANDB_API_KEY", "my-api-key")
    sm.get("WANDB_API_KEY")  # "my-api-key"

    # On TPU VM: load secrets as env vars
    tpu.load_secrets(["WANDB_API_KEY", "HF_TOKEN"])
"""

import subprocess
import json


class SecretManager:
    """
    Manage secrets in Google Cloud Secret Manager.

    Secrets are stored server-side and accessed by the TPU VM via IAM —
    they never pass through the local machine or appear in command history.
    """

    def __init__(self, project=None):
        self.project = project or self._get_project()

    @staticmethod
    def _get_project():
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True,
        )
        proj = result.stdout.strip()
        if not proj or proj == "(unset)":
            raise RuntimeError("No gcloud project set. Run: gcloud config set project PROJECT")
        return proj

    def _gcloud(self, args, check=True, timeout=30):
        cmd = ["gcloud"] + args + [f"--project={self.project}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if check and result.returncode != 0:
            raise RuntimeError(f"gcloud error: {result.stderr.strip()}")
        return result

    # ----------------------------------------------------------------
    # CRUD
    # ----------------------------------------------------------------
    def create(self, name, value):
        """
        Create or update a secret.

        Args:
            name: Secret name (e.g. "WANDB_API_KEY")
            value: Secret value
        """
        # Check if exists
        if self.exists(name):
            # Add new version
            proc = subprocess.Popen(
                ["gcloud", "secrets", "versions", "add", name,
                 f"--project={self.project}", "--data-file=-"],
                stdin=subprocess.PIPE, capture_output=True, text=True,
            )
            proc.communicate(input=value)
            if proc.returncode != 0:
                raise RuntimeError(f"Failed to update secret: {proc.stderr}")
        else:
            # Create new secret
            proc = subprocess.Popen(
                ["gcloud", "secrets", "create", name,
                 f"--project={self.project}", "--data-file=-"],
                stdin=subprocess.PIPE, capture_output=True, text=True,
            )
            proc.communicate(input=value)
            if proc.returncode != 0:
                raise RuntimeError(f"Failed to create secret: {proc.stderr}")
        print(f"Secret '{name}' stored in Cloud Secret Manager")

    def get(self, name, version="latest"):
        """Read a secret value."""
        result = self._gcloud([
            "secrets", "versions", "access", version,
            f"--secret={name}",
        ])
        return result.stdout

    def exists(self, name):
        """Check if a secret exists."""
        result = self._gcloud(
            ["secrets", "describe", name, "--format=json"],
            check=False,
        )
        return result.returncode == 0

    def delete(self, name):
        """Delete a secret."""
        self._gcloud(["secrets", "delete", name, "--quiet"])

    def list(self):
        """List all secrets."""
        result = self._gcloud(["secrets", "list", "--format=json"])
        secrets = json.loads(result.stdout)
        return [s["name"].split("/")[-1] for s in secrets]

    # ----------------------------------------------------------------
    # Grant TPU VM access
    # ----------------------------------------------------------------
    def grant_tpu_access(self, secret_name, service_account=None):
        """
        Grant the TPU VM's service account access to a secret.

        Args:
            secret_name: Name of the secret
            service_account: SA email (default: auto-detect compute engine default)
        """
        if service_account is None:
            result = self._gcloud([
                "iam", "service-accounts", "list",
                "--format=value(email)",
                "--filter=displayName:Compute Engine default",
            ])
            service_account = result.stdout.strip()
            if not service_account:
                raise RuntimeError("Could not find default compute service account")

        self._gcloud([
            "secrets", "add-iam-policy-binding", secret_name,
            f"--member=serviceAccount:{service_account}",
            "--role=roles/secretmanager.secretAccessor",
        ])
        print(f"Granted {service_account} access to '{secret_name}'")

    def grant_tpu_access_all(self, service_account=None):
        """Grant TPU VM access to ALL secrets."""
        for name in self.list():
            try:
                self.grant_tpu_access(name, service_account)
            except Exception as e:
                print(f"  Warning: {name}: {e}")

    # ----------------------------------------------------------------
    # Remote loading (run on TPU VM)
    # ----------------------------------------------------------------
    @staticmethod
    def load_env_command(secret_names, project=None):
        """
        Generate a shell command that loads secrets as env vars on the TPU VM.
        The VM uses its service account — no local secrets involved.

        Args:
            secret_names: List of secret names to load
            project: GCP project (auto-detected if None)

        Returns:
            Shell command string to run on the VM
        """
        proj = f"--project={project}" if project else ""
        cmds = []
        for name in secret_names:
            cmds.append(
                f'export {name}=$(gcloud secrets versions access latest --secret={name} {proj})'
            )
        return " && ".join(cmds)
