# Security Policy

## Supported Versions

We actively support the following versions of QMANN with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

The QMANN team takes security seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do Not Create a Public Issue

Please **do not** create a public GitHub issue for security vulnerabilities. This could put users at risk.

### 2. Report Privately

Send an email to **info@neuraparse.com** (or contact the maintainers directly) with:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any suggested fixes (if you have them)

### 3. Response Timeline

- **Initial Response**: We will acknowledge receipt within 48 hours
- **Assessment**: We will assess the vulnerability within 5 business days
- **Fix Timeline**: Critical vulnerabilities will be addressed within 7 days, others within 30 days
- **Disclosure**: We will coordinate with you on responsible disclosure

### 4. What to Include

When reporting a vulnerability, please include:

- **Type of vulnerability** (e.g., code injection, privilege escalation, etc.)
- **Location** (file path, function name, line number if possible)
- **Impact** (what an attacker could achieve)
- **Reproduction steps** (detailed steps to reproduce)
- **Proof of concept** (if applicable and safe to share)
- **Suggested mitigation** (if you have ideas)

## Security Considerations for QMANN

### Quantum-Specific Security

QMANN involves quantum computing components that have unique security considerations:

1. **Quantum State Privacy**: Quantum states may contain sensitive information
2. **Circuit Tampering**: Malicious modification of quantum circuits
3. **Side-Channel Attacks**: Information leakage through quantum measurements
4. **Classical-Quantum Interface**: Security at the boundary between classical and quantum components

### Classical Security

Standard security practices apply to the classical components:

1. **Input Validation**: All user inputs are validated
2. **Dependency Security**: Regular updates of dependencies
3. **Code Injection**: Protection against malicious code execution
4. **Data Privacy**: Secure handling of training data and model parameters

### Research Data Security

Since QMANN is a research project:

1. **Experimental Data**: Secure storage and transmission of research data
2. **Model Weights**: Protection of trained model parameters
3. **Reproducibility**: Ensuring security doesn't compromise reproducibility
4. **Collaboration**: Secure sharing of research artifacts

## Security Best Practices for Users

### Installation Security

1. **Verify Downloads**: Always download from official sources
2. **Check Signatures**: Verify package signatures when available
3. **Use Virtual Environments**: Isolate QMANN installations
4. **Regular Updates**: Keep QMANN and dependencies updated

### Usage Security

1. **Input Sanitization**: Validate all data inputs to QMANN models
2. **Access Control**: Limit access to quantum hardware credentials
3. **Network Security**: Use secure connections for remote quantum backends
4. **Logging**: Monitor and log quantum circuit executions

### Development Security

1. **Code Review**: All contributions undergo security review
2. **Static Analysis**: Regular security scanning of codebase
3. **Dependency Scanning**: Automated vulnerability detection in dependencies
4. **Secure Defaults**: Security-first default configurations

## Known Security Considerations

### Current Limitations

1. **Quantum Simulation**: Classical simulation may leak quantum state information
2. **Hardware Access**: Quantum hardware access requires credential management
3. **Network Communication**: Communication with quantum backends over networks
4. **Experimental Nature**: Research code may have undiscovered vulnerabilities

### Mitigation Strategies

1. **Encryption**: Sensitive data is encrypted at rest and in transit
2. **Access Controls**: Role-based access to quantum resources
3. **Audit Logging**: Comprehensive logging of security-relevant events
4. **Regular Reviews**: Periodic security assessments

## Security Updates

Security updates will be:

1. **Prioritized**: Security fixes take precedence over feature development
2. **Documented**: Clear documentation of what was fixed
3. **Communicated**: Users notified through multiple channels
4. **Tested**: Thoroughly tested before release

## Contact Information

For security-related questions or concerns:

- **Email**: info@neuraparse.com
- **Organization**: Neura Parse (@neuraparse)
- **Website**: https://neuraparse.com
- **GPG Key**: [Link to public key if available]

## Acknowledgments

We appreciate the security research community's efforts to improve QMANN's security. Researchers who responsibly disclose vulnerabilities will be acknowledged (with their permission) in our security advisories.

## Legal

This security policy is subject to our [Code of Conduct](CODE_OF_CONDUCT.md) and [Contributing Guidelines](CONTRIBUTING.md). By participating in our security process, you agree to these terms.

---

**Last Updated**: July 2025
**Version**: 1.0
