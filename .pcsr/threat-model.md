# Threat model

The same model, in Threat Composer format, is in `threat-model.tc.json`. The trust zones and flows are described in
`architecture.md`.

## Scope

An engineer runs this sample on one ephemeral EC2 Neuron instance in their own AWS account, by hand on a Neuron Deep
Learning AMI or with `scripts/launch_ec2.sh`. The sample creates no service, endpoint or data store. The only
resource it creates besides the instance is the security group `neuron-spectral-dns` (no ingress, egress TCP 443
only). The optional instance profile and results bucket belong to the caller.

Assets: the caller's AWS credentials and account, the instance and its compute time, the integrity of the code the
instance runs as root at boot, and the run output (benchmark CSVs and logs, not sensitive).

## Trust zones

| zone | trust |
|---|---|
| TZ001 Engineer workstation (AWS CLI, own credentials) | high |
| TZ002 AWS regional endpoints (SSM Parameter Store, EC2 API, Session Manager, results bucket) | high |
| TZ003 Default VPC public subnet (instance, root volume, instance profile, internet gateway) | medium |
| TZ004 Internet (public source repository) | untrusted |

## Assumptions

| # | assumption |
|---|---|
| A1 | The engineer runs the sample in a non-production account with their own least-privilege credentials, as listed in `docs/launcher-policy.json`. |
| A2 | The Neuron Deep Learning AMI and the Neuron SDK it carries (NKI, neuronx-cc, libnrt) are trusted AWS software. |
| A3 | The engineer reviews the tag or commit they run; the default is the release tag of the public repository, served over HTTPS. |
| A4 | No customer or regulated data is processed. |
| A5 | Each run uses one ephemeral instance; hosting the solver as a multi-tenant service is out of scope. |
| A6 | The optional instance profile and results bucket belong to the caller and are configured by the caller. |

## Mitigations

| # | mitigation |
|---|---|
| M1 | Dedicated security group `neuron-spectral-dns`: no ingress rule, egress TCP 443 only. The launcher refuses any group, including one passed with `-G`, that has an ingress rule. No key pair; nothing listens. |
| M2 | Interactive access only through Session Manager, tied to an IAM identity; no SSH, no open port. |
| M3 | IMDSv2 required at launch (`HttpTokens=required`). |
| M4 | Least-privilege instance role: `AmazonSSMManagedInstanceCore` plus `s3:PutObject` on one prefix. Without a profile the instance has no AWS permission. |
| M5 | Pinned source: the git path fetches exactly one tag or commit (`-c`, default the release tag) and logs the resolved commit SHA. A tarball source is refused without its SHA-256. |
| M6 | Automatic teardown: shutdown behaviour "terminate" plus `shutdown -h +TTL` in user data (1 to 1440 minutes, default 60). |
| M7 | Output goes only to the prefix named with `-u`, with the caller's own role; the sample creates no bucket and ships no bucket policy. NEFFs are not uploaded. |
| M8 | No secrets in the tree: account, region, profile and bucket are caller-supplied parameters; the tree is secret-scanned before publication. |
| M9 | Encrypted gp3 root volume, deleted on termination. |
| M10 | HTTPS for every transfer; the DLAMI id comes from the public AWS-owned SSM parameter, read with the caller's IAM credentials. |
| M11 | Kernels run in the Neuron runtime on a single-tenant instance; the C drivers use the documented libnrt API, check `mmap`, `fopen`, allocations and exact input file sizes, bound integer arguments, and are built with `-fstack-protector-strong -D_FORTIFY_SOURCE=2`. Nothing is installed at run time. |
| M12 | Every Session Manager session and EC2 API call is attributable to an IAM principal in CloudTrail. |
| M13 | Least-privilege launcher policy (`docs/launcher-policy.json`): `ssm:GetParameter` on the Neuron DLAMI parameters only, the security-group actions, `RunInstances`, `CreateTags` only on create, and `iam:PassRole` on one role with `iam:PassedToService` `ec2.amazonaws.com`. |
| M14 | Launcher input validation: region, instance type, TTL, grid, ranks, source (https only, no shell metacharacters), ref, SHA-256, security group id, profile and S3 prefix are checked against strict patterns before they reach the root user-data script. |
| M15 | Checked tarball: `-S <sha256>` is required and verified with `sha256sum -c` before extraction. A presigned URL is scoped to one object with a short expiry, because user data is readable with `ec2:DescribeInstanceAttribute`. |

## Threats

| # | threat | STRIDE | priority | mitigations | residual |
|---|---|---|---|---|---|
| T1 | An internet actor opens an inbound connection to the instance's public IP to reach a shell, the runtime or the code. | E | Medium | M1, M2 | Low |
| T2 | A process on the instance reads the optional role credentials from instance metadata and uses them outside the run. | I | High | M3, M4, M1 | Low |
| T3 | An actor who controls the source repository serves tampered or unreviewed code that the instance runs as root. | T | High | M5, M4, M10 | Low for the default release tag; a caller who passes another source or ref owns that choice |
| T4 | An operator forgets a costly instance, which keeps running. | D | Medium | M6 | Low |
| T5 | An operator names an unintended or public bucket with `-u`, disclosing the run output. | I | Low | M7, M4, M9 | Low |
| T6 | A contributor publishes account ids, bucket names, instance ids or tokens with the sample. | I | High | M8 | Low |
| T7 | A malicious or faulty kernel corrupts device state or tries to escape the instance. | E | Low | M11 | Low |
| T8 | A compromised dependency in the AMI's Neuron SDK or NumPy injects code into the build or the run. | T | Medium | M11, M5 | Medium: rests on the AMI (A2) |
| T9 | A network attacker spoofs the DLAMI id or the repository to serve a malicious image or source. | S | Medium | M10 | Low |
| T10 | An operator acts on the instance through a channel no principal can be held to account for. | R | Low | M12, M2 | Low |
| T11 | A holder of broad launcher credentials passes a more privileged role to the instance, changes existing resources or launches in an unintended place. | E | Medium | M13, M12 | Low with the documented policy |
| T12 | An actor who controls a launcher argument injects shell commands into the user-data script that runs as root. | T | Medium | M14 | Low |
| T13 | An actor substitutes a tampered source tarball, or reuses a presigned URL read from user data. | T, I | Medium | M15, M14 | Low; an unexpired presigned URL stays readable to principals allowed `ec2:DescribeInstanceAttribute` |
| T14 | A workload that shares a security group with the instance reaches it through the group's member-to-member rule. | E | Medium | M1 | Low: the instance's group has no ingress rule, including self-references |

## Residual risk

The main residual risks are the trust placed in the AMI and its Neuron SDK (T8), and the choice of source when an
engineer overrides the default release tag (T3, T13). Egress on TCP 443 goes to any address, so code already running
on the instance could send data out; the instance holds no secret and processes no customer data, and without a
profile it has no AWS permission. Costs of the instance are the caller's; an inf2.24xlarge is about 7 USD per hour.

## Out of scope

Multi-tenant hosting of the solver as a service, production use, and any data other than the benchmark case.
