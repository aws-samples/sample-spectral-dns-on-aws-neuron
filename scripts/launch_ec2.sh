#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Launch one Neuron instance that fetches a pinned revision of this repository and runs scripts/bootstrap.sh from user
# data, then powers itself off after TTL minutes (the instance is launched with shutdown behaviour "terminate").
#   scripts/launch_ec2.sh [-r region] [-t instance-type] [-m ttl-minutes] [-g grid] [-R ranks]
#                         [-s source] [-c ref] [-S sha256] [-G sg-id] [-p instance-profile] [-u s3://bucket/prefix]
# -s/-c: an https git URL and the tag or commit to fetch (default: the public repository at the release tag).
# -s/-S: an https URL to a .tar.gz of this tree with one top-level directory, and its SHA-256, checked before
#        extraction (for testing a tree that is not published; a presigned URL is readable in the instance's user data
#        by anyone allowed ec2:DescribeInstanceAttribute until it expires).
# -G:    an existing security group to use. Without it the script uses, or creates once per VPC, the security group
#        "neuron-spectral-dns": no ingress rule, egress TCP 443 only. Any group used must have no ingress rule.
# -p/-u: with an instance profile, reach the box through Session Manager; with -u, copy build/ (logs and CSVs, not the
#        NEFFs) to <prefix>/<run-id>/ when the run ends (the profile needs s3:PutObject on that prefix).
# The AMI is the latest Neuron JAX Deep Learning AMI (Ubuntu 24.04) of the region, read from its public SSM parameter.
# The root volume is encrypted with the account's default EBS key and deleted with the instance; metadata is IMDSv2 only.
# docs/launcher-policy.json lists the IAM permissions this script needs.
set -euo pipefail
RELEASE_REF=v1.0.0
REGION=eu-north-1; TYPE=inf2.xlarge; TTL=60; GRID=64; RANKS=1; PROFILE=""; UPLOAD=""; SG=""; SHA=""; REF=""
SRC="https://github.com/aws-samples/sample-spectral-dns-on-aws-neuron.git"
while getopts "r:t:m:g:R:s:c:S:G:p:u:" o; do case $o in
  r) REGION=$OPTARG;; t) TYPE=$OPTARG;; m) TTL=$OPTARG;; g) GRID=$OPTARG;; R) RANKS=$OPTARG;; s) SRC=$OPTARG;;
  c) REF=$OPTARG;; S) SHA=$OPTARG;; G) SG=$OPTARG;; p) PROFILE=$OPTARG;; u) UPLOAD=$OPTARG;; *) exit 2;; esac; done

die(){ echo "launch_ec2.sh: $*" >&2; exit 2; }
ok(){ [[ $1 =~ $2 ]] || die "invalid $3: $1"; }
ok "$REGION" '^[a-z]{2}(-[a-z]+)+-[0-9]$' region
ok "$TYPE" '^[a-z0-9]+\.[a-z0-9]+$' instance-type
ok "$TTL" '^[0-9]{1,4}$' ttl-minutes; (( TTL >= 1 && TTL <= 1440 )) || die "ttl-minutes must be 1..1440"
ok "$GRID" '^[0-9]{2,3}$' grid
ok "$RANKS" '^[0-9]{1,2}$' ranks
ok "$SRC" '^https://[A-Za-z0-9._~:/?#@!&()*+,;=%-]+$' source
[ -z "$PROFILE" ] || ok "$PROFILE" '^[A-Za-z0-9+=,.@_-]{1,128}$' instance-profile
[ -z "$UPLOAD" ] || ok "$UPLOAD" '^s3://[a-z0-9.-]+(/[A-Za-z0-9._/-]*)?$' upload-prefix
[ -z "$SG" ] || ok "$SG" '^sg-[0-9a-f]{8,17}$' security-group
if [ -n "$SHA" ]; then
  ok "$SHA" '^[0-9a-f]{64}$' sha256; [ -z "$REF" ] || die "-c and -S are exclusive (git ref or tarball checksum)"
else
  case "$SRC" in *.tgz|*.tgz\?*|*.tar.gz|*.tar.gz\?*) die "a tarball source needs its SHA-256 (-S)";; esac
  REF=${REF:-$RELEASE_REF}; ok "$REF" '^[A-Za-z0-9._/-]{1,100}$' ref
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-n${GRID}-r${RANKS}"
unset AWS_REGION AWS_DEFAULT_REGION
AMI=$(aws ssm get-parameter --region "$REGION" --name /aws/service/neuron/dlami/jax-0.10/ubuntu-24.04/latest/image_id --query Parameter.Value --output text)
VPC=$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
[ "$VPC" != None ] || die "no default VPC in $REGION"
if [ -z "$SG" ]; then
  SG=$(aws ec2 describe-security-groups --region "$REGION" --filters Name=group-name,Values=neuron-spectral-dns Name=vpc-id,Values="$VPC" --query 'SecurityGroups[0].GroupId' --output text)
  if [ "$SG" = None ]; then
    SG=$(aws ec2 create-security-group --region "$REGION" --vpc-id "$VPC" --group-name neuron-spectral-dns \
      --description "neuron-spectral-dns: no ingress, egress TCP 443 only" \
      --tag-specifications 'ResourceType=security-group,Tags=[{Key=project,Value=neuron-spectral-dns}]' --query GroupId --output text)
    aws ec2 revoke-security-group-egress --region "$REGION" --group-id "$SG" --ip-permissions 'IpProtocol=-1,IpRanges=[{CidrIp=0.0.0.0/0}]' > /dev/null
    aws ec2 authorize-security-group-egress --region "$REGION" --group-id "$SG" --ip-permissions 'IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges=[{CidrIp=0.0.0.0/0,Description=HTTPS}]' > /dev/null
    echo "created security group $SG (delete it with: aws ec2 delete-security-group --region $REGION --group-id $SG)"
  fi
fi
INGRESS=$(aws ec2 describe-security-groups --region "$REGION" --group-ids "$SG" --query 'length(SecurityGroups[0].IpPermissions)' --output text)
[ "$INGRESS" = 0 ] || die "security group $SG has $INGRESS ingress rule(s); the instance needs none"

if [ -n "$SHA" ]; then
  FETCH="curl -fsSL -o src.tar.gz '$SRC'
echo '$SHA  src.tar.gz' | sha256sum -c -
tar xzf src.tar.gz --strip-components=1 -C neuron-spectral-dns && rm src.tar.gz
echo 'source tarball sha256 $SHA'"
else
  FETCH="git -C neuron-spectral-dns init -q
git -C neuron-spectral-dns fetch -q --depth 1 '$SRC' '$REF'
git -C neuron-spectral-dns checkout -q FETCH_HEAD
echo \"source $SRC ref $REF commit \$(git -C neuron-spectral-dns rev-parse HEAD)\""
fi
UD=$(mktemp); cat > "$UD" <<UDEOF
#!/usr/bin/env bash
shutdown -h +$TTL "neuron-spectral-dns TTL"
exec > /home/ubuntu/user-data.log 2>&1
set -euo pipefail
cd /home/ubuntu && mkdir neuron-spectral-dns
$FETCH
chown -R ubuntu:ubuntu neuron-spectral-dns
cd neuron-spectral-dns
setsid nohup bash -c 'sudo -u ubuntu env GRID=$GRID RANKS=$RANKS bash scripts/bootstrap.sh; if [ -n "$UPLOAD" ]; then aws s3 cp --region $REGION --recursive --exclude "*.neff" --exclude "*.bin" --exclude "*/art-*" build "$UPLOAD/$RUN_ID/" --only-show-errors; fi' > /home/ubuntu/bootstrap.out 2>&1 < /dev/null &
UDEOF
IP=(); [ -n "$PROFILE" ] && IP=(--iam-instance-profile "Name=$PROFILE")
aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" --security-group-ids "$SG" ${IP[@]+"${IP[@]}"} \
  --instance-initiated-shutdown-behavior terminate --metadata-options HttpTokens=required,HttpEndpoint=enabled \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,Encrypted=true,DeleteOnTermination=true}' \
  --user-data "file://$UD" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=neuron-spectral-dns-n${GRID}-r${RANKS}},{Key=project,Value=neuron-spectral-dns},{Key=ttl-min,Value=$TTL}]" \
  --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone,LaunchTime]' --output text
echo "run-id $RUN_ID security-group $SG"
rm -f "$UD"
