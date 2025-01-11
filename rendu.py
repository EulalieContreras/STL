import marimo

__generated_with = "0.9.17"
app = marimo.App(width="medium")


@app.cell
def __():
    # Python Standard Library
    import json

    # Marimo
    import marimo as mo

    # Third-Party Librairies
    import numpy as np
    import matplotlib.pyplot as plt
    import mpl3d
    from mpl3d import glm
    from mpl3d.mesh import Mesh
    from mpl3d.camera import Camera

    import meshio

    np.seterr(over="ignore")  # 🩹 deal with a meshio false warning

    import sdf
    from sdf import sphere, box, cylinder
    from sdf import X, Y, Z
    from sdf import intersection, union, orient, difference

    def show(
        filename,
        theta=0.0,
        phi=0.0,
        scale=1.0,
        colormap="viridis",
        edgecolors=(0, 0, 0, 0.25),
        figsize=(6, 6),
    ):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1], xlim=[-1, +1], ylim=[-1, +1], aspect=1)
        ax.axis("off")
        camera = Camera("ortho", theta=theta, phi=phi, scale=scale)
        mesh = meshio.read(filename)
        vertices = glm.fit_unit_cube(mesh.points)
        faces = mesh.cells[0].data
        vertices = glm.fit_unit_cube(vertices)
        mesh = Mesh(
            ax,
            camera.transform,
            vertices,
            faces,
            cmap=plt.get_cmap(colormap),
            edgecolors=edgecolors,
        )
        return mo.center(fig)
    return (
        Camera,
        Mesh,
        X,
        Y,
        Z,
        box,
        cylinder,
        difference,
        glm,
        intersection,
        json,
        meshio,
        mo,
        mpl3d,
        np,
        orient,
        plt,
        sdf,
        show,
        sphere,
        union,
    )


@app.cell
def __(show):
    show("data/cube.stl", theta=45.0, phi=30.0, scale=1)
    return


@app.cell
def __(show):
    show("data/cube.stl", theta=315.0, phi=330.0, scale=1)
    return


@app.cell
def __(np):
    def make_STL(triangles, normals=None, name=""):
        stl = f'solid {name}'
        for i in range (len(triangles)):
            if type(normals) == type(None):
                normal = np.cross(triangles[i, 1]-triangles[i, 0], triangles[i, 2]-triangles[i, 0])
                stl = stl + f'\n  facet normal {normal[0]} {normal[1]} {normal[2]}'
            else: 
                stl = stl + f'\n  facet normal {normals[i, 0]} {normals[i, 1]} {normals[i, 2]}'

            point = f'\n      vertex {triangles[i, 0, 0]} {triangles[i, 0, 1]} {triangles[i, 0, 2]}' + f'\n      vertex {triangles[i, 1, 0]} {triangles[i, 1, 1]} {triangles[i, 1, 2]}' + f'\n      vertex {triangles[i, 2, 0]} {triangles[i, 2, 1]} {triangles[i, 2, 2]}'
            stl = stl + '\n    outer loop' + point + '\n    endloop'
            stl = stl + '\n  endfacet'
        stl = stl + f'\nendsolid {name}'
        return stl
    return (make_STL,)


@app.cell
def __(make_STL, np):
    square_triangles = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    make_STL(square_triangles, name="square")
    return (square_triangles,)


@app.cell
def __(np):
    def tokenize(stl):
        l = stl.split()
        for i in range(len(l)):
            try:
                l[i] = np.float32(l[i])
            except:
                pass
        return l
    return (tokenize,)


@app.cell
def __(tokenize):
    with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file:
        square_stl = square_file.read()
    tokens = tokenize(square_stl)
    print(tokens)
    return square_file, square_stl, tokens


@app.cell
def __(np):
    def parse(tokens):
        triangle = []
        normal = []
        name = tokens[1]
        t = -1

        for i in range(len(tokens)):
            if tokens[i] == 'normal':
                t += 1
                normal.append([tokens[i+1], tokens[i+2], tokens[i+3]])
                triangle.append([])
            if tokens[i] == 'vertex':
                triangle[t].append([tokens[i+1], tokens[i+2], tokens[i+3]]) 

        triangles = np.array(triangle, dtype= np.float32)
        normals = np.array(normal, dtype= np.float32)

        return triangles, normals, name
    return (parse,)


@app.cell
def __(parse, tokenize):
    with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file1:
        square_stl1 = square_file1.read()
    tokens1 = tokenize(square_stl1)
    triangles1, normals1, name1 = parse(tokens1)
    print(repr(triangles1))
    print(repr(normals1))
    print(repr(name1))
    return name1, normals1, square_file1, square_stl1, tokens1, triangles1


@app.cell
def __(np, parse, tokenize):
    def positive_octant(solide):
        tokens = tokenize(solide)
        triangles, normals, name = parse(tokens)
        violations = 0
        for i in range(len(triangles)):
            for j in range(len(triangles[i])):
                if triangles[i, j, 0] < 0 or triangles[i, j, 1] < 0 or triangles[i, j, 2] < 0:
                    violations += 1
        pourcentage = (violations / (3*len(triangles))) * 100
        return violations == 0, pourcentage

    def orientation(solide):
        tokens = tokenize(solide)
        triangles, normals, name = parse(tokens)
        violations = 0
        for i in range(len(normals)):
            if np.dot(normals[i], np.cross(triangles[i, 1]-triangles[i, 0], triangles[i, 2]-triangles[i, 0])) or (normals[0]**2 + normals[1]**2 + normals[2]**2)==1 :
                violations += 1
        pourcentage = (violations / len(normals)) * 100
        return violations == 0, pourcentage

    def shared_edge(solide):
        tokens = tokenize(solide)
        triangles, normals, name = parse(tokens)
        dict = {}
        for t in triangles:
            s = np.abs(t[1]-t[0])
            s = tuple(s)
            if s in dict:
                dict[s] += 1
            else:
                dict[s] = 1
            s = np.abs(t[2]-t[1])
            s = tuple(s)
            if s in dict:
                dict[s] += 1
            else:
                dict[s] = 1
            s = np.abs(t[2]-t[0])
            s = tuple(s)
            if s in dict:
                dict[s] += 1
            else:
                dict[s] = 1
        violations = 0
        for i in dict:
            if dict[i] != 2:
                violations += 1
        pourcentage = (violations / (3*len(triangles)))*100
        return violations == 0, pourcentage

    def ascending(solide):
        tokens = tokenize(solide)
        triangles, normals, name = parse(tokens)
        barycentres = []
        for t in triangles:
            barycentre = np.mean([t[0], t[1], t[2]], axis=0)
            barycentres.append(barycentre[2])
        violations = 0
        for i in range(len(triangles)-1):
            if barycentres[i] > barycentres[i+1]:
                violations += 1
        pourcentage = (violations / len(triangles)) * 100
        return violations == 0, pourcentage

    def diagnostic(solide):
        resultat = not(positive_octant(solide)[0]) and not(orientation(solide)[1] == 0) and shared_edge(solide)[0] and not(ascending(solide)[0])
        liste = [positive_octant(solide)[1], orientation(solide)[1], shared_edge(solide)[1], ascending(solide)[1]]
        return resultat, liste
    return ascending, diagnostic, orientation, positive_octant, shared_edge


@app.cell
def __(diagnostic):
    with open("data/cube.stl", mode="rt", encoding="us-ascii") as cube_file:
        cube = cube_file.read()
    diagnostic(cube)
    return cube, cube_file


@app.cell
def __(make_STL, np):
    def OBJ_to_STL(obj):
        liste = obj.split('\n')
        liste = liste[3:]
        triangles = []
        for i in range(len(liste)):
            if len(liste[i]) != 0 and liste[i][0] == 'f':
                triangle = liste[i].split()
                triangle.pop(0)
                for k in range(3):
                    triangle[k] = float(triangle[k])
                sommets = []
                for j in range(3):
                    s = liste[j].split()
                    s.pop(0)
                    for k in range(3):
                        s[k] = float(s[k])
                    sommets.append(s)
                triangles.append(sommets)
        solide = np.array(triangles)
        return make_STL(solide)
    return (OBJ_to_STL,)


@app.cell
def __(OBJ_to_STL):
    with open("data/bunny.obj", mode="rt", encoding="us-ascii") as bunny_file:
        bunny = bunny_file.read()
    bunny_stl = OBJ_to_STL(bunny)
    bunny_stl
    return bunny, bunny_file, bunny_stl


@app.cell
def __(make_STL, np):
    def STL_binary_to_text(stl_filename_in, stl_filename_out):
        with open(stl_filename_in, mode="rb") as file:
            _ = file.read(80)
            n = np.fromfile(file, dtype=np.uint32, count=1)[0]
            normals = []
            faces = []
            for i in range(n):
                normals.append(np.fromfile(file, dtype=np.float32, count=3))
                faces.append(np.fromfile(file, dtype=np.float32, count=9).reshape(3, 3))
                _ = file.read(2)
        face = np.array(faces)
        normal = np.array(normals)
        stl_text = make_STL(face, normal)
        with open(stl_filename_out, mode="wt", encoding="utf-8") as file:
            file.write(stl_text)
    return (STL_binary_to_text,)


@app.cell
def __(STL_binary_to_text):
    STL_binary_to_text('data\dragon.stl', 'data\dragon_ascii.stl')
    return


@app.cell
def __():
    import jupytercad
    dir(jupytercad)
    #forme = sphere(1) & box(1.5)
    #cylindre = cylinder(0.5)
    #forme = forme - (cylinder(X) | cylinder(Y) | cylinder(Z))

    #shapes ou primitives n'apparaissent pas dans dir() car j'ai un problème de version de jupytercad que je n'arrive pas à résoudre 
    return (jupytercad,)


app._unparsable_cell(
    r"""
    def jcad_to_stl(file_in, file_out):
        with open(file_in, mode=\"rb\") as file:
        _ = file.read(80)
        n = np.fromfile(file, dtype=np.uint32, count=1)[0]
        normals = []
        faces = []
        for i in range(n):
            normals.append(np.fromfile(file, dtype=np.float32, count=3))
            faces.append(np.fromfile(file, dtype=np.float32, count=9).reshape(3, 3))
            _ = file.read(2)
        face = np.array(faces)
        normal = np.array(normals)
        stl_text = make_STL(face, normal)
        with open(file_out, mode=\"wt\", encoding=\"utf-8\") as file:
            file.write(stl_text)
    """,
    name="__"
)


if __name__ == "__main__":
    app.run()
