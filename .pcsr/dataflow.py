# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Generate .pcsr/dataflow-diagram.png: data flows between the trust zones of running this sample. Needs the `graphviz`
Python package (installed with `diagrams`) and graphviz. Run from the repository root:
    python3 .pcsr/dataflow.py"""
import os

import graphviz

HERE = os.path.dirname(os.path.abspath(__file__))

DOT = r'''
digraph dfd {
  graph [dpi=150];
  graph [rankdir=LR, fontname="Helvetica", fontsize=22, labelloc=t, pad=0.5, nodesep=0.8, ranksep=1.6,
         newrank=true, label="neuron-spectral-dns: data flows between trust zones\n\n"];
  node  [fontname="Helvetica", fontsize=13, style=filled, fillcolor=white];
  edge  [fontname="Helvetica", fontsize=12];

  subgraph cluster_tz001 {
    label="TZ001  Engineer workstation (trust: high)"; labeljust=l; fontsize=15;
    style="rounded,dashed"; color="#666666"; bgcolor="#f4f4f4";
    engineer [label="Engineer\nAWS CLI, least-privilege\ncredentials\n(docs/launcher-policy.json)", fillcolor="#dbe8f8"];
  }

  subgraph cluster_tz002 {
    label="TZ002  AWS regional endpoints, caller's account (trust: high)"; labeljust=l; fontsize=15;
    style="rounded,dashed"; color="#3f7fbf"; bgcolor="#eaf2fb";
    param [label="SSM Parameter Store\n(public DLAMI id)"];
    ec2api [label="EC2 API"];
    ssm [label="Session Manager\n(optional, IAM)"];
    s3 [label="Results S3 prefix\n(D001, caller's own,\noptional)", shape=cylinder, fillcolor="#d5e8d4"];
  }

  subgraph cluster_tz003 {
    label="TZ003  Default VPC public subnet, caller's account (trust: medium)\nsecurity group neuron-spectral-dns: no ingress, egress TCP 443 only";
    labeljust=l; labelloc=b; fontsize=15; style="rounded,dashed"; color="#6f9f4f"; bgcolor="#eef6e8";
    role [label="Instance profile\n(optional, least privilege)"];
    box [label="Neuron EC2 instance\n(inf2 / trn1)\nIMDSv2 required, no key pair,\nterminates after TTL"];
    ebs [label="Encrypted root EBS\n(D002, deleted on\ntermination)", shape=cylinder, fillcolor="#d5e8d4"];
    igw [label="Internet gateway\n(egress path, public IP)"];
  }

  subgraph cluster_tz004 {
    label="TZ004  Internet (trust: untrusted)"; labeljust=l; fontsize=15;
    style="rounded,dashed"; color="#bf3f3f"; bgcolor="#fbeaea";
    repo [label="Public source repository\nrelease tag / commit\n(read only)", shape=box, fillcolor="#f8cecc"];
  }

  engineer -> param  [label="1  ssm:GetParameter (DLAMI id)\nHTTPS, IAM  [CP001]\n "];
  engineer -> ec2api [label="2  security group, RunInstances, iam:PassRole\nvalidated user data: ref, GRID, RANKS, TTL\nHTTPS, IAM  [CP001]\n "];
  engineer -> ssm    [label="6  ssm:StartSession (optional)\nHTTPS, IAM  [CP001]\n ", style=dashed, color="#1f6fbf", fontcolor="#1f6fbf"];
  ec2api -> box      [label="3  launch instance\n[CP002]\n "];
  s3 -> box          [dir=back, label="5  s3:PutObject logs + CSVs (optional)\nHTTPS 443, IAM  [CP003]\n ", style=dashed, color="#1f6fbf", fontcolor="#1f6fbf"];
  ssm -> box         [dir=back, label="SSM agent, outbound HTTPS 443\n(optional)  [CP003]\n ", style=dashed, color="#1f6fbf", fontcolor="#1f6fbf"];
  role -> box        [label="credentials via IMDSv2\n(intra-zone)\n ", style=dotted, color="#8f5fa0", fontcolor="#8f5fa0"];
  box -> ebs         [label="source + build artefacts\n(intra-zone)\n ", style=dotted, color="#6f9f4f", fontcolor="#6f9f4f"];
  box -> igw         [label="4a  egress only, HTTPS 443  [CP004]\n ", color="#bf3f3f", fontcolor="#bf3f3f"];
  igw -> repo        [label="4b  git fetch pinned ref, or\nSHA-256-checked tarball\nHTTPS 443  [CP004]\n ", color="#bf3f3f", fontcolor="#bf3f3f"];

  { rank=same; param; ec2api; ssm; s3; }
  { rank=same; ebs; igw; }
}
'''

graphviz.Source(DOT).render(outfile=os.path.join(HERE, "dataflow-diagram.png"), format="png",
                             engine="dot", cleanup=True)
