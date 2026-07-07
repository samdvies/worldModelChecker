#!/usr/bin/env bash
# Launches a spot (or --on-demand) g4dn.xlarge GPU box to populate the
# pretrained-encoder (DINOv2 / V-JEPA-2) latent caches. Run FROM the laptop.
#
# Usage:
#   scripts/aws/launch_gpu.sh [--on-demand] [--dry-run]
#
# --dry-run prints every AWS CLI call instead of executing it, but still
# runs `aws sts get-caller-identity` for real as a credential sanity check.
#
# The box gets a public IP (needed for internet egress in the default VPC,
# which has no NAT gateway), but isolation comes from a zero-inbound-rule
# security group plus SSM-only access -- no open ingress ports. It
# self-terminates via `remote_job.sh` (or the 240-minute cost backstop
# inside it) on completion or failure.
set -euo pipefail

PROFILE="claude-admin"
REGION="eu-west-1"
ACCOUNT_ID="652742769396"
BUCKET="physics-auditor-${ACCOUNT_ID}"
PROJECT_TAG="physics-auditor"
INSTANCE_TYPE="g4dn.xlarge"
ROOT_VOLUME_GB=100
IAM_ROLE_NAME="physics-auditor-gpu-role"
IAM_PROFILE_NAME="physics-auditor-gpu-profile"
SG_NAME="physics-auditor-gpu-sg"

ON_DEMAND=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --on-demand) ON_DEMAND=true ;;
        --dry-run) DRY_RUN=true ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# aws_cmd: in --dry-run mode, print the command instead of running it.
aws_cmd() {
    if [ "$DRY_RUN" = true ]; then
        printf '[dry-run] aws'
        printf ' %q' "$@"
        printf '\n'
    else
        aws "$@"
    fi
}

echo "== credential check (real, even in --dry-run) =="
aws sts get-caller-identity --profile "$PROFILE" --region "$REGION"

echo "== 1. tar repo (git archive HEAD) =="
ARCHIVE="$(mktemp -d)/physics-auditor.tar.gz"
if [ "$DRY_RUN" = true ]; then
    echo "[dry-run] git -C $REPO_ROOT archive --format=tar.gz -o $ARCHIVE HEAD"
else
    git -C "$REPO_ROOT" archive --format=tar.gz -o "$ARCHIVE" HEAD
fi

echo "== 2. ensure S3 bucket $BUCKET exists (idempotent) =="
aws_cmd s3api create-bucket \
    --bucket "$BUCKET" \
    --create-bucket-configuration "LocationConstraint=${REGION}" \
    --profile "$PROFILE" --region "$REGION" \
    || echo "(bucket may already exist -- continuing)"

echo "== 3. upload repo archive + remote_job.sh to S3 =="
aws_cmd s3 cp "$ARCHIVE" "s3://${BUCKET}/repo/physics-auditor.tar.gz" \
    --profile "$PROFILE" --region "$REGION"
aws_cmd s3 cp "$SCRIPT_DIR/remote_job.sh" "s3://${BUCKET}/scripts/remote_job.sh" \
    --profile "$PROFILE" --region "$REGION"

echo "== 4. IAM role + instance profile (idempotent, SSM + scoped S3 access) =="
TRUST_POLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
S3_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::${BUCKET}", "arn:aws:s3:::${BUCKET}/*"]
    }
  ]
}
EOF
)

aws_cmd iam create-role \
    --role-name "$IAM_ROLE_NAME" \
    --assume-role-policy-document "$TRUST_POLICY" \
    --profile "$PROFILE" \
    || echo "(role may already exist -- continuing)"

aws_cmd iam attach-role-policy \
    --role-name "$IAM_ROLE_NAME" \
    --policy-arn "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore" \
    --profile "$PROFILE"

aws_cmd iam put-role-policy \
    --role-name "$IAM_ROLE_NAME" \
    --policy-name "physics-auditor-s3-scoped" \
    --policy-document "$S3_POLICY" \
    --profile "$PROFILE"

aws_cmd iam create-instance-profile \
    --instance-profile-name "$IAM_PROFILE_NAME" \
    --profile "$PROFILE" \
    || echo "(instance profile may already exist -- continuing)"

aws_cmd iam add-role-to-instance-profile \
    --instance-profile-name "$IAM_PROFILE_NAME" \
    --role-name "$IAM_ROLE_NAME" \
    --profile "$PROFILE" \
    || echo "(role may already be attached -- continuing)"

echo "== 5. resolve latest PyTorch Deep Learning AMI via SSM public parameter =="
AMI_PARAM="/aws/service/deeplearning/ami/x86_64/pytorch/latest/ami-id"
if [ "$DRY_RUN" = true ]; then
    aws_cmd ssm get-parameters --names "$AMI_PARAM" --profile "$PROFILE" --region "$REGION"
    AMI_ID="ami-dryrun00000000000"
else
    AMI_ID=$(aws ssm get-parameters --names "$AMI_PARAM" \
        --query "Parameters[0].Value" --output text \
        --profile "$PROFILE" --region "$REGION")
fi
echo "AMI: $AMI_ID"

echo "== 6. build user-data (downloads + runs remote_job.sh from S3) =="
USER_DATA=$(cat <<EOF
#!/bin/bash
set -e
mkdir -p /opt/physics-auditor
aws s3 cp "s3://${BUCKET}/scripts/remote_job.sh" /opt/physics-auditor/remote_job.sh --region ${REGION}
chmod +x /opt/physics-auditor/remote_job.sh
BUCKET="${BUCKET}" REGION="${REGION}" /opt/physics-auditor/remote_job.sh
EOF
)
USER_DATA_B64=$(printf '%s' "$USER_DATA" | base64 -w0 2>/dev/null || printf '%s' "$USER_DATA" | base64)

echo "== 7. ensure security group $SG_NAME exists (idempotent, no inbound rules) =="
if [ "$DRY_RUN" = true ]; then
    aws_cmd ec2 describe-vpcs --filters Name=isDefault,Values=true \
        --query "Vpcs[0].VpcId" --output text \
        --profile "$PROFILE" --region "$REGION"
    DEFAULT_VPC_ID="vpc-dryrun00000000000"
    aws_cmd ec2 describe-security-groups \
        --filters "Name=group-name,Values=${SG_NAME}" \
        --query "SecurityGroups[0].GroupId" --output text \
        --profile "$PROFILE" --region "$REGION"
    aws_cmd ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "physics-auditor GPU box: no inbound, SSM-only access" \
        --vpc-id "$DEFAULT_VPC_ID" \
        --profile "$PROFILE" --region "$REGION"
    SG_ID="sg-dryrun00000000000"
else
    DEFAULT_VPC_ID=$(aws ec2 describe-vpcs \
        --filters Name=isDefault,Values=true \
        --query "Vpcs[0].VpcId" --output text \
        --profile "$PROFILE" --region "$REGION")
    SG_ID=$(aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=${SG_NAME}" \
        --query "SecurityGroups[0].GroupId" --output text \
        --profile "$PROFILE" --region "$REGION" 2>/dev/null || true)
    if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
        SG_ID=$(aws ec2 create-security-group \
            --group-name "$SG_NAME" \
            --description "physics-auditor GPU box: no inbound, SSM-only access" \
            --vpc-id "$DEFAULT_VPC_ID" \
            --query "GroupId" --output text \
            --profile "$PROFILE" --region "$REGION")
    fi
fi
echo "SG: $SG_ID"

echo "== 8. request instance ($([ "$ON_DEMAND" = true ] && echo on-demand || echo spot)) =="
BLOCK_DEVICE_MAPPINGS='[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":'"$ROOT_VOLUME_GB"',"VolumeType":"gp3"}}]'
TAG_SPEC='ResourceType=instance,Tags=[{Key=Project,Value='"$PROJECT_TAG"'}]'

RUN_INSTANCE_ARGS=(
    ec2 run-instances
    --image-id "$AMI_ID"
    --instance-type "$INSTANCE_TYPE"
    --iam-instance-profile "Name=${IAM_PROFILE_NAME}"
    --block-device-mappings "$BLOCK_DEVICE_MAPPINGS"
    --instance-initiated-shutdown-behavior terminate
    --tag-specifications "$TAG_SPEC"
    --user-data "$USER_DATA_B64"
    --associate-public-ip-address
    --security-group-ids "$SG_ID"
    --profile "$PROFILE" --region "$REGION"
)

if [ "$ON_DEMAND" = false ]; then
    RUN_INSTANCE_ARGS+=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"InstanceInterruptionBehavior":"terminate"}}')
fi

aws_cmd "${RUN_INSTANCE_ARGS[@]}"

echo "== done. Use pull_results.sh once the job finishes, or terminate manually: =="
echo "  aws ec2 terminate-instances --instance-ids <id> --profile $PROFILE --region $REGION"
