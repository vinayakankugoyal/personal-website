---
layout: doc
---

# Vinayak Goyal

**Email:** vinayak@purelydysfunctional.com  
**LinkedIn:** [vinayakgoyal](https://linkedin.com/in/vinayakgoyal)  
**GitHub:** [vinayakankugoyal](https://github.com/vinayakankugoyal)

---

## Work Experience

### **Google** | Staff Software Engineer
*Kubernetes Security Team | Nov 2024 - Present*

As the technical lead I architected the roadmap and provided leadership for a 7 member software and security engineering team focused on hardening Google Kubernetes Engine (GKE) and OSS Kubernetes.

- Proposed, designed and led a team to build a token broker service for GKE system workloads to exchange pod service account tokens for down-scoped IAM service account tokens for metrics and logging. This allowed us to decouple GKE system workload permissions from customer workload permissions and the GKE node service account. Giving us the flexibility to add new capabilities to system workloads without requiring any actions from customers.
- Proposed, designed and implemented [KEP-4633](https://kep.k8s.io/4633) in OSS Kubernetes. This KEP allowed admins to explicitly configure which endpoints support anonymous authentication. This prevents cluster exploitation even if a user mistakenly created RBAC bindings for system:anonymous. To raise awareness about this security misconfiguration and about this KEP I gave a [talk](https://www.youtube.com/watch?v=PbZbojx4kVM) at KubeCon NA.
- Designed and implemented [KEP-2862](https://kep.k8s.io/2862) in OSS Kubernetes. This KEP breaks the often misused nodes/proxy permissions into more fine-grained permissions. Allowing users to write RBAC policies that only grant the permissions required by their workloads, in a way that was backwards compatible. Before this KEP permissions to access the /healthz endpoint and /exec/ required the same nodes/proxy subresource permission.
- Proposed, designed and implemented [KEP-5040](https://kep.k8s.io/5040) in OSS Kubernetes. This KEP removed support for the built-in gitRepo volume driver which had been unmaintained. Recently vulnerabilities were found which meant Kubernetes and Cloud Providers would spend precious resources patching it. I worked with SIG-STORAGE and SIG-API to drive consensus in removing this volume driver from Kubernetes.

### **Google** | Senior Software Engineer
*Kubernetes Security Team | Nov 2021 - Nov 2024*

- Led a team of SWEs and SEs defining the roadmap and OKRs and identifying the top x risks and prioritizing work that mitigated the risk.
- Spearheaded the design, implementation and deployment of the Kubernetes Security Validation service, a proprietary policy engine for scanning misconfigurations in Kubernetes workloads. This system proactively detects thousands of Kubernetes security misconfigurations across GKE and high-priority Google 1Ps, and prevents such workloads from being submitted; producing an inventory of the existing risk. Led a multi-year cross functional program to remediate the findings from the service and reduced them from thousands to a few well documented exemptions.
- io_uring was the largest contributor to GKE kernel vulnerabilities and Google spent upwards of $1.8M in bug bounties on it. I worked with containerd maintainers and drove consensus to block io_using system calls in the default seccomp profile, a change that docker also made following our lead.
- Designed an allowlist system for popular partner workloads to be onboarded to GKE Autopilot clusters. GKE Autopilot clusters apply restrictions on workloads that prevent popular logging, monitoring and security scanning tools like jFrog, CrowdStrike, Wiz etc. from running on these clusters. I built an allowlisting scheme that matched the incoming workload to the allowlist and allowed it to bypass certain restrictions when the match succeeded.

### **Google** | Software Engineer
*Kubernetes Security Team | Sep 2019 - Nov 2021*

- Proposed the migration of GKE workloads to rootless. I owned this workstream end-to-end working with teams across the GKE org and migrated all GKE system workloads to rootless with a few well documented exemptions. This vastly reduced the risk of container escapes in GKE and its customers. I proposed [KEP-2568](https://kep.k8s.io/2568) and drove consensus in Kubernetes to support running the Kubernetes control-plane components as non-root in kubeadm, which is the canonical tool used to deploy Kubernetes clusters by millions. I also gave a [talk](https://www.youtube.com/watch?v=uouH9fsWVIE) at KubeCon about the importance and techniques of how to migrate container workloads to rootless.
- Designed and implemented secure by default controls for GKE Autopilot clusters that automatically applied security controls on customer workloads like dropping CAP_NET_RAW and applying seccomp filters to disallow dangerous syscalls.
- Designed and implemented CMEK for boot disk and etcd secrets encryption for GKE Autopilot clusters. This allowed customers to encrypt the node bootdisk and Kubernetes secrets stored in etcd with a key that was managed by customers putting them in control of their data.

### **Microsoft** | Software Engineer
*Office Security Team | 2016 - 2019*

- Developed and launched Microsoft [Safelinks](https://docs.microsoft.com/en-us/office365/securitycompliance/atp-safe-links), a critical security feature that safeguards millions of Office 365 users by seamlessly integrating real-time link reputation checks across all Office applications.
- Invented and implemented the [Security Policy Advisor](https://docs.microsoft.com/en-us/DeployOffice/overview-of-security-policy-advisor) (US Patent 10,454,952 B2), an intelligent service that leverages organizational usage patterns to generate data-driven security policy recommendations for tenant administrators.

### **Microsoft** | Software Engineering Intern
*Office PowerPoint Team | Summer 2015*

- Enhanced PowerPoint's user experience by implementing and refining Smart Guides, significantly improving the ease and precision with which users can align and snap shapes.
- Experimented 3D Printing functionality in PowerPoint by developing and implementing an .obj exporter capable of processing entire slideshows and preparing 3D shapes for physical printing.

---

## Education

### **University Of Southern California**
**M.S. in Computer Science** | *2014 - 2016*

**Relevant Coursework:**
- Operating Systems (CSCI402)
- Advanced Operating Systems (CSCI555)
- Advanced Distributed Systems (CSCI657)

**Activities:**
- Trojan Judo Club

### **Thapar University**
**B.E. in Computer Science and Engineering** | *2010 - 2014*

**Relevant Coursework:**
- Computer Networking
- Computer Architecture
- Object Oriented Programming in C++
- Introduction to Programming with C

**Activities:**
- IM Soccer
- IM Basketball

---

## Skills

### Leadership & Strategy
- Technical Strategy & Roadmapping
- Cross-functional Leadership
- Engineering Mentorship
- Open Source Governance

### Core Competencies
- System Design
- Distributed Systems
- Cloud Security
- Container Security
- Kernel Development
- Virtualization

### Programming Languages
- **Proficient:** C, C++, Rust, Go, Python, JavaScript
- **Familiar:** OCaml, Lean, Kotlin

### Technologies & Frameworks
- **Backend:** Node.js, Express, FastAPI, gRPC, WebSockets
- **Frontend:** React, Next.js
- **Databases:** PostgreSQL, MongoDB, Redis, Supabase
- **Security & Identity:** OPA, SPIFFE/SPIRE, OAuth/OIDC, Reverse Engineering, Network Forensics
- **DevOps & Cloud:** Kubernetes, Docker, AWS, GCP, Azure, Terraform, Nix, GitHub Actions, Prometheus, Grafana

---

## Talks

### KubeCon EU 2023
**Least Privilege Containers: Keeping a Bad Day from Getting Worse**
Video : https://www.youtube.com/watch?v=uouH9fsWVIE

### KubeCon NA 2023
**RBACdoors: How Cryptominers Are Exploiting RBAC Misconfigs**
Video: https://www.youtube.com/watch?v=PbZbojx4kVM