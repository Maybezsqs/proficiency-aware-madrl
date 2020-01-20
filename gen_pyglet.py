import pyglet
fname = 'ksu.png'
window = pyglet.window.Window()
#image = pyglet.resource.image(fname)
image = pyglet.image.load(fname)

#@window.event
def on_draw():
    window.clear()
    image.blit(300, 400)

pyglet.app.run()
