# mlops-bikeshare
## Quickstart

```powershell
# Windows PowerShell
$env:AWS_PROFILE="Shirley"
$env:AWS_REGION="ca-central-1"

# Bootstrap backend once
aws s3api create-bucket --bucket mlops-tfstate-387706002632-ca-central-1 --create-bucket-configuration LocationConstraint=$env:AWS_REGION
aws dynamodb create-table --table-name mlops-tflock --attribute-definitions AttributeName=LockID,AttributeType=S --key-schema AttributeName=LockID,KeyType=HASH --billing-mode PAY_PER_REQUEST

cd infra/terraform
terraform init -reconfigure
terraform apply
