from mako.template import Template

with open("docs/README-tmpl.md", "r") as f:
    tmpl_txt = f.read()
tmpl_txt = tmpl_txt.replace("##", "${'##'}")
tmpl = Template(tmpl_txt)

result = tmpl.render()
with open("README.md", "w") as f:
    f.write(result)
