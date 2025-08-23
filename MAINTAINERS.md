# Maintainer Guide for Shinkansen Passenger Satisfaction Predictor

This document provides essential guidance for repository maintainers, including instructions for enabling branch protection rules that cannot be configured programmatically via API.

## 🔒 Branch Protection Setup

### Overview

Branch protection rules help ensure code quality by requiring specific checks before code can be merged into protected branches. Since GitHub's REST API cannot enable all protection features, these must be configured manually through the web interface.

### Step-by-Step Instructions

#### 1. Navigate to Repository Settings

1. Go to the repository: `https://github.com/kebrahimpour/shinkansen-passenger-satisfaction-predictor`
2. Click on the **Settings** tab (requires admin access)
3. In the left sidebar, click on **Branches**

#### 2. Add Branch Protection Rule

1. Click **Add rule** button
2. In the "Branch name pattern" field, enter: `main`
3. Configure the following protection settings:

##### Required Status Checks

☑️ **Require status checks to pass before merging**
- ☑️ Require branches to be up to date before merging
- ☑️ **Select the following status checks:**
  - `CI / test (3.10)` - Python 3.10 test suite
  - `CI / test (3.11)` - Python 3.11 test suite  
  - `CI / test (3.12)` - Python 3.12 test suite
  - `Lint` - Code quality checks
  - `Test` - Legacy test workflow (if present)
  - `Build` - Build verification

##### Pull Request Requirements

☑️ **Require a pull request before merging**
- ☑️ Require approvals: **1** (minimum)
- ☑️ Dismiss stale reviews when new commits are pushed
- ☑️ Require review from CODEOWNERS (if applicable)

##### Additional Restrictions

☑️ **Restrict pushes that create files larger than 100 MB**

☑️ **Require linear history**

☑️ **Include administrators** (applies rules to admins too)

☑️ **Allow force pushes** → **Everyone** (for emergency fixes)

☑️ **Allow deletions** → **Disabled**

4. Click **Create** to save the branch protection rule

### 3. Verification Steps

After setting up branch protection, verify the configuration:

1. Go to **Settings** → **Branches**
2. Confirm the `main` branch shows protection rules
3. Test with a sample PR to ensure:
   - Status checks are required
   - Reviews are enforced
   - Direct pushes to main are blocked

## 🔧 Maintenance Tasks

### Regular Maintenance

#### Weekly
- Review and merge approved pull requests
- Monitor CI/CD pipeline health
- Update dependencies via Dependabot PRs
- Review and address security alerts

#### Monthly
- Review branch protection effectiveness
- Update documentation as needed
- Analyze code quality metrics
- Plan releases and version tags

#### Quarterly
- Review and update testing matrix
- Evaluate new Python versions for support
- Update development tools and pre-commit hooks
- Security audit of dependencies

### Emergency Procedures

#### Critical Bug Fixes

1. **Immediate Fix Required:**
   - Create hotfix branch from main
   - Apply minimal fix with tests
   - Fast-track review process
   - Merge with expedited approval

2. **Security Vulnerabilities:**
   - Follow responsible disclosure
   - Create private security advisory
   - Develop fix in private repository
   - Coordinate public release

#### CI/CD Pipeline Issues

1. **Pipeline Failures:**
   - Investigate failing checks
   - Temporary disable problematic checks if needed
   - Fix underlying issues
   - Re-enable all checks

2. **Infrastructure Problems:**
   - Monitor GitHub Actions status
   - Check third-party service availability
   - Implement fallback procedures
   - Document incidents for future reference

## 👥 Team Management

### Access Levels

- **Admin**: Repository owners and lead maintainers
- **Maintain**: Core contributors with merge rights
- **Write**: Regular contributors with push access to branches
- **Triage**: Community members who can manage issues/PRs
- **Read**: All community members

### Adding New Maintainers

1. Evaluate contributor history and expertise
2. Discuss with existing maintainer team
3. Invite via repository settings
4. Provide maintainer onboarding documentation
5. Grant appropriate access level

### Removing Access

1. Review need for access removal
2. Coordinate with team lead
3. Remove access via repository settings
4. Update team documentation
5. Secure any shared credentials

## 📊 Monitoring and Metrics

### Key Metrics to Track

- **Code Quality:**
  - Test coverage percentage
  - Code review completion rate
  - Security vulnerability count
  - Technical debt metrics

- **Community Health:**
  - Contributor activity
  - Issue response time
  - PR merge time
  - Community feedback scores

- **CI/CD Performance:**
  - Build success rate
  - Test execution time
  - Deployment frequency
  - Mean time to recovery

### Monitoring Tools

- **GitHub Insights**: Repository analytics and community metrics
- **Actions Dashboard**: CI/CD pipeline monitoring
- **Security Tab**: Vulnerability and security alerts
- **Dependabot**: Automated dependency updates

## 📝 Release Management

### Release Process

1. **Pre-Release:**
   - Update CHANGELOG.md
   - Verify all tests pass
   - Update version numbers
   - Create release branch

2. **Release:**
   - Create GitHub release with tag
   - Generate release notes
   - Publish to package registries
   - Update documentation

3. **Post-Release:**
   - Monitor for issues
   - Address community feedback
   - Plan next release cycle
   - Update project roadmap

### Version Strategy

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## 🔮 Troubleshooting Common Issues

### Branch Protection Not Working

**Symptoms:** Direct pushes to main still work

**Solution:**
1. Check admin enforcement settings
2. Verify status check names match CI jobs
3. Ensure all required checks are configured
4. Confirm user permissions are correct

### CI Checks Always Failing

**Symptoms:** Required status checks never complete

**Solution:**
1. Verify workflow file syntax
2. Check repository secrets and permissions
3. Review GitHub Actions logs
4. Test workflow on feature branch

### Performance Issues

**Symptoms:** Slow CI/CD pipeline execution

**Solution:**
1. Optimize test execution
2. Use build matrix efficiently
3. Cache dependencies
4. Parallelize independent jobs

## 📩 Contact Information

### Primary Maintainer
- **Name**: Keyvan Ebrahimpour
- **GitHub**: [@kebrahimpour](https://github.com/kebrahimpour)
- **Email**: k1.ebrahimpour@gmail.com

### Escalation Contacts
- **Security Issues**: Create private security advisory
- **Infrastructure**: GitHub Support
- **Community Issues**: Repository discussions

## 📚 Additional Resources

- [GitHub Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Semantic Versioning Guidelines](https://semver.org/)
- [Conventional Commits Specification](https://www.conventionalcommits.org/)

---

**Last Updated**: August 2025  
**Next Review**: November 2025

> ⚠️ **Important**: Keep this document updated as the project evolves. Review and update quarterly or after significant changes to the repository structure or team composition.
