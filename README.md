![Alt text](/graphic_app/asteroids_logo_1.png?raw=true "Logo")

# Social
[![Discord](https://img.shields.io/badge/discord-asteroidsathome-blue?logo=discord)](https://discord.gg/PDd5gkyJ4f)

# Asteroids@home

Project Website: https://asteroidsathome.net/

Asteroids@home is a volunteer distributed computing project developed at the [Astronomical Institute](http://astro.troja.mff.cuni.cz/index_en.html), [Charles University in Prague](http://www.cuni.cz/UKENG-1.html), in cooperation with [Radim Vančo](http://www.czechnationalteam.cz/?q=content/profily-kyong) from [CzechNationalTeam](http://www.czechnationalteam.cz/). The project is directed by [Josef Durech](http://www.mff.cuni.cz/toUTF8.en/fakulta/struktura/lide/2968.htm). It runs on the Berkeley Open Infrastructure for Network Computing ([BOINC](http://boinc.berkeley.edu/)) software platform and uses power of volunteers' computers to solve the lightcurve inversion problem for many asteroids.

## Why distributed computing?
With huge amount of photometric data coming from big all-sky surveys as well as from backyard astronomers, the lightcurve inversion becomes a computationaly demanding process. In the future, we can expect even more data from surveys that are either already operating ([PanSTARRS](http://pan-starrs.ifa.hawaii.edu/public/)) or under construction ([Gaia](http://www.esa.int/export/esaSC/120377_index_0_m.html), [LSST](http://www.lsst.org/lsst/)). Moreover, data from surveys are often sparse in time, which means that the rotation period - the basic physical parameter - cannot be estimated from the data easily. Contrary to classical lightcurves where the period is "visible" in the data, a wide interval of all possible periods has to be scanned densely when analysing sparse data. This fact enormously enlarges the computational time and the only practical way to efficiently handle photometry of hundreds of thousands of asteroids is to use distributed computing. Moreover, the problem is ideal for parallelization - the period interval can be divided into smaller parts that are searched separately and then the results are joined together.

## Why to study asteroids?
The large discrepancy between the huge number of all known asteroids and the small number of those with known basic physical parameters (shape, spin, period) is a strong motivation for further research.
Knowing the physical properties of a significant part of the asteroid population is necessary for understanding the origin and evolution of the whole solar system.
Thermal emission of small asteroids can significantly change their orbit ([Yarkovsky efect](http://en.wikipedia.org/wiki/Yarkovsky_effect)), which can be crucial for predicting the probability of their collision with the Earth. To be able to compute how the thermal emission affects the orbit, we have to know the spin (and also the shape, to a certain extent) of the object.
Scientific objectives
The aim of the project is to derive shapes and spin for a significant part of the asteroid population. As input data, we use any asteroid photometry that is available. The results are asteroid convex shape models with the direction of the spin axis and the rotation period. The models will be published in peer-reviewed journals and then made public in the [DAMIT](https://astro.troja.mff.cuni.cz/projects/damit/) database.

### Note
The [Astronomical Institute](http://astro.troja.mff.cuni.cz/index_en.html) at [Charles University in Prague](http://www.cuni.cz/UKENG-1.html) holds the copyright on all Asteroids@home applications source code. By submitting contributions to the Asteroids@home code, you irrevocably assign all right, title, and interest, including copyright and all copyright rights, in such contributions to The Regents of the The [Astronomical Institute](http://astro.troja.mff.cuni.cz/index_en.html) at [Charles University in Prague](http://www.cuni.cz/UKENG-1.html), who may then use the code for any purpose that it desires.

## Reporting Security Issues
Please report security issues by emailing
[Radim Vančo](mailto:radim.vanco@jifox.cz?subject=[Asteroids@home]%20Security%20Issues).

# License
Asteroids@home is free software; you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Asteroids@home applications are distributed in the hope that they will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with BOINC.  If not, see <https://www.gnu.org/licenses/>.