# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Generate .pcsr/architecture.png: trust zones and data flows of running this sample. Needs the `diagrams` package
and graphviz. Run from the repository root:
    python3 .pcsr/architecture.py"""
import os
from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2, EC2Instance
from diagrams.aws.general import User
from diagrams.aws.management import SystemsManager, SystemsManagerParameterStore
from diagrams.aws.network import InternetGateway
from diagrams.aws.security import IAMRole
from diagrams.aws.storage import S3, EBS
from diagrams.onprem.vcs import Github

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = {"fontsize": "20", "bgcolor": "white", "pad": "0.6", "ranksep": "2.2", "nodesep": "0.7", "splines": "spline"}
NODE = {"fontsize": "13"}
EDGE = {"fontsize": "12"}
ZONE = lambda color, fill, style="rounded", loc="t": {"color": color, "bgcolor": fill, "style": style, "fontsize": "14",
                                                      "margin": "20", "labelloc": loc,
                                                      "labeljust": "r" if loc == "b" else "l"}
API = {"color": "#333333"}
OPT = {"color": "#1f5f9f", "fontcolor": "#1f5f9f", "style": "dashed"}
NET = {"color": "#c0392b", "fontcolor": "#c0392b"}

with Diagram("neuron-spectral-dns: how the sample runs (trust zones and data flows)", show=False,
             filename=os.path.join(HERE, "architecture"), outformat="png", direction="LR",
             graph_attr=GRAPH, node_attr=NODE, edge_attr=EDGE):
    with Cluster("Engineer workstation (trusted by the engineer)", graph_attr=ZONE("#888888", "#f4f4f4")):
        user = User("Engineer: AWS CLI with own\nleast-privilege credentials\n(docs/launcher-policy.json)")

    with Cluster("Caller's AWS account", graph_attr=ZONE("#999999", "white")):
        with Cluster("AWS regional endpoints (HTTPS, IAM-authenticated)", graph_attr=ZONE("#6f8fbf", "#eef3fd")):
            param = SystemsManagerParameterStore("Public SSM parameter\n(Neuron DLAMI id)")
            ec2api = EC2("EC2 API")
            ssm = SystemsManager("Session Manager\n(optional)")
            s3 = S3("Results bucket, caller's own\n(optional, one prefix)")

        with Cluster("Default VPC, public subnet", graph_attr=ZONE("#7fae6f", "#f0fbef")):
            role = IAMRole("Instance profile (optional)\nSSM core + s3:PutObject\non one prefix")
            with Cluster("Security group \"neuron-spectral-dns\"\nno ingress, egress TCP 443 only",
                         graph_attr=ZONE("#5f9f4f", "#e0f3dc", "dashed", "b")):
                box = EC2Instance("Neuron instance (inf2 / trn1)\nIMDSv2 required, no key pair,\n"
                                  "terminates after TTL")
                vol = EBS("Root volume\nencrypted, deleted\non termination")
            igw = InternetGateway("Internet gateway\n(public IP, egress path)")

    with Cluster("Internet (untrusted)", graph_attr=ZONE("#c0392b", "#fdeeee", "dashed")):
        repo = Github("Public repository\nrelease tag / commit\n(read only)")

    user >> Edge(label="1  ssm:GetParameter\n\n", **API) >> param
    user >> Edge(label="2  security group, RunInstances,\niam:PassRole; validated user data\n\n", **API) >> ec2api
    user >> Edge(label="6  ssm:StartSession (optional)\n\n", **OPT) >> ssm
    ec2api >> Edge(label="3  launch instance", **API) >> box
    box >> Edge(style="dotted", color="#5f9f4f", arrowhead="none") >> vol
    role >> Edge(label="credentials via IMDSv2\n\n", color="#7d3c98", fontcolor="#7d3c98", style="dashed") >> box
    box >> Edge(label="4  git fetch pinned ref or\nSHA-256-checked tarball\nHTTPS 443", **NET) >> igw
    igw >> Edge(label="egress only\n\n", **NET) >> repo
    # Drawn from the service to the instance so the endpoints stay left of the VPC; dir=back points the arrow outbound.
    s3 >> Edge(label="5  s3:PutObject logs + CSVs\nHTTPS 443 (optional)", dir="back", **OPT) >> box
    ssm >> Edge(label="SSM agent, HTTPS 443\n(optional)", dir="back", **OPT) >> box
