# Pull Request

## 📋 Summary

<!-- Provide a brief description of what this PR does -->

## 🎯 Type of Change

<!-- Mark the relevant option with an "x" -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not
  work as expected)
- [ ] 📚 Documentation update
- [ ] 🧹 Code refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] 🎨 Style/formatting changes
- [ ] 🔧 Configuration changes
- [ ] 🧪 Test improvements

## 🔗 Related Issues

<!-- Link to any related issues -->

Closes #<!-- issue number -->

## 📝 Detailed Description

### What changed?

<!-- Describe the changes in detail -->

### Why was this change made?

<!-- Explain the motivation behind the changes -->

### How was it implemented?

<!-- Briefly describe the implementation approach -->

## 🧪 Testing

<!-- Describe how you tested your changes -->

- [ ] Unit tests pass locally (`uv run pytest`)
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Added new tests for the changes

### Test Commands

<!-- List the commands you ran to test -->

```bash
uv run pytest
uv run ruff check .
uv run mypy modules
```

## 📸 Screenshots/Examples

<!-- If applicable, add screenshots or code examples -->

<details>
<summary>Before/After Comparison</summary>

**Before:**

```python
# old code
```

**After:**

```python
# new code
```

</details>

## 🚀 Performance Impact

<!-- If applicable, describe any performance implications -->

- [ ] No performance impact
- [ ] Performance improvement: <!-- describe -->
- [ ] Performance regression: <!-- describe and justify -->

## 📋 Checklist

### Code Quality

- [ ] I have run `uv run pre-commit run --all-files`
- [ ] I have run `uv run pytest` and all tests pass
- [ ] I have run `uv run ruff check .` and `uv run ruff format .`
- [ ] I have run `uv run mypy modules` and resolved type issues
- [ ] My code follows the project's style guidelines

### Documentation

- [ ] I have updated docstrings for new/modified functions
- [ ] I have updated the README.md if needed
- [ ] I have updated CLAUDE.md if this affects AI development workflows
- [ ] I have added/updated examples if applicable

### Testing

- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally
- [ ] I have added integration tests if applicable

### Dependencies

- [ ] I have updated `pyproject.toml` if I added new dependencies
- [ ] I have run `uv lock` to update the lock file
- [ ] New dependencies are necessary and well-justified

## 🔄 Breaking Changes

<!-- If this is a breaking change, describe the impact and migration path -->

- [ ] This PR introduces breaking changes
- [ ] Migration guide is provided
- [ ] Version bump is appropriate

## 👀 Reviewer Notes

<!-- Any specific areas you'd like reviewers to focus on -->

## 📚 Additional Context

<!-- Add any other context about the PR here -->

______________________________________________________________________

**AI-Assisted Development Notice:**

- [ ] This PR was developed with Claude Code assistance
- [ ] AI-generated code has been reviewed and tested
- [ ] Documentation reflects AI development workflow if applicable
