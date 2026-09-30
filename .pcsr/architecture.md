# Architecture and trust boundaries

This sample deploys no service. It is source code an engineer runs on one EC2 Neuron instance (inf2 or trn1) in
their own account, either by hand (fetch the repository on a Neuron Deep Learning AMI and run
`scripts/bootstrap.sh`) or with `scripts/launch_ec2.sh`, which launches the instance with user data that fetches a
pinned revision and runs the same script. Two diagrams show the design:

- `architecture.png` (generator `architecture.py`): the components and trust zones.
- `dataflow-diagram.png` (generator `dataflow.py`): the numbered flows and the crossing points between zones.

Regenerate both from the repository root with `python3 .pcsr/architecture.py` (needs the `diagrams` package) and
`python3 .pcsr/dataflow.py` (needs graphviz).

## Trust zones

| zone | contents | trust |
|---|---|---|
| TZ001 Engineer workstation | AWS CLI with the engineer's own least-privilege credentials (`docs/launcher-policy.json`) | high |
| TZ002 AWS regional endpoints | SSM Parameter Store, EC2 API, Session Manager, the caller's optional results bucket | high |
| TZ003 Default VPC, public subnet | the Neuron instance, its encrypted root volume, the optional instance profile, the internet gateway | medium |
| TZ004 Internet | the public source repository | untrusted |

## Crossings and their controls

| crossing | flow | control |
|---|---|---|
| CP001 TZ001 to TZ002 | 1 `ssm:GetParameter` for the Neuron JAX DLAMI id; 2 security-group calls, `ec2:RunInstances`, `iam:PassRole`; 6 optional `ssm:StartSession` | HTTPS and IAM. The launcher policy grants `ssm:GetParameter` on `/aws/service/neuron/dlami/*` only, the security-group actions, `RunInstances`, `CreateTags` only on create, and `iam:PassRole` on one role with `iam:PassedToService` `ec2.amazonaws.com`. Every launcher input is checked against a strict pattern before it reaches the root user-data script. |
| CP002 TZ002 to TZ003 | 3 launch | IMDSv2 required (`HttpTokens=required`), no key pair, encrypted gp3 root volume deleted on termination, shutdown behaviour "terminate" plus `shutdown -h +TTL` (1 to 1440 minutes, default 60). The security group has no ingress rule; the launcher refuses any group that has one. |
| CP003 TZ003 to TZ002 | 5 optional `s3:PutObject` of logs and CSVs to `s3://bucket/prefix/<run-id>/`; SSM agent outbound | HTTPS 443 and the caller's own instance profile: `AmazonSSMManagedInstanceCore` plus `s3:PutObject` on one prefix. NEFFs are not uploaded. Without a profile the instance has no AWS permission. |
| CP004 TZ003 to TZ004 | 4a egress through the internet gateway; 4b source fetch | Egress TCP 443 only, read only. The git path fetches exactly one tag or commit (`-c`, default the release tag) with `git fetch --depth 1` and `git checkout FETCH_HEAD`, and logs the resolved commit SHA. The tarball path requires `-S <sha256>`, checked with `sha256sum -c` before extraction. |

Inside TZ003 the instance reads the optional role credentials from IMDSv2 (link-local) and writes source and build
artefacts to its root volume. DNS resolution uses the Amazon VPC resolver, which security groups do not filter.

## Network exposure

Nothing in the sample listens on the network. The instance has a public IP in the default subnet only so that it can
reach the source repository and, when used, S3 and Session Manager through the internet gateway. Its dedicated
security group, `neuron-spectral-dns`, has no ingress rule and allows egress on TCP 443 only, so no connection can be
made to the instance, from the internet or from other workloads in the VPC. `launch_ec2.sh` creates this group once
per VPC, tagged `project=neuron-spectral-dns`. Access, when wanted, goes through Session Manager (IAM), not SSH.

Nothing is installed at run time: the Neuron SDK and NumPy come from the AMI.

## Instance contents

The instance holds no secret. The user data carries the source URL and ref (or a tarball URL and its SHA-256), the
grid size, the rank count, the TTL and the optional S3 prefix. A presigned tarball URL is readable in user data by
anyone with `ec2:DescribeInstanceAttribute` until it expires, so it is scoped to one object with a short expiry.

The C drivers check `mmap`, `fopen` and allocation failures and exact input file sizes, bound their integer
arguments, and are built with `-fstack-protector-strong -D_FORTIFY_SOURCE=2`.

## Data

The instance processes no customer data. The input is the analytic Taylor-Green field computed on the box. The
output is a CSV of kinetic energy, enstrophy and dissipation per time step, plus logs. The reference CSVs in
`references/` are public benchmark numbers.

## Publicly exposed vs private

| exposed to the internet | private |
|---|---|
| nothing listens; the instance reads the public source repository over HTTPS | the EC2 instance (public IP, no ingress rule; Session Manager only) |
| | the optional results bucket (caller's own, IAM-authenticated writes from the instance role) |
| | instance metadata (IMDSv2 tokens required) |
